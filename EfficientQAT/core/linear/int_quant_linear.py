from typing import List, Dict, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantizer import(
    UniformAffineQuantizer,
    GradualQuantizer,
    QuantConfig
)

# 因为涉及到的 weight quantizer 有不同类型
def build_weight_quantizer(
    prefix: str,
    weight: torch.Tensor,
    config: QuantConfig,
) -> Union[UniformAffineQuantizer, GradualQuantizer]:
    """
    根据配置构建并返回相应的权重量化器实例

    Args:
        prefix (str): 量化器前缀标识符
        weight (torch.Tensor): 待量化的权重张量
        config (QuantConfig): 量化配置对象，包含量化类型等参数

    Returns:
        BaseQuantizer: 具体量化器实例，可能是UniformAffineQuantizer或GradualQuantizer

    Raises:
        ValueError: 当config.quant_type不是支持的量化类型时
    """
    if config.quant_type == "uniform_affine":
        return UniformAffineQuantizer(prefix, weight, config)
    elif config.quant_type == "gradual":
        return GradualQuantizer(prefix, weight, config)
    else:
        # fallback to uniform_affine 如果是无设置初始化
        # prefix 应该要能够通过上下文推测出来
        # weight和config应该是默认能够设置的
        return UniformAffineQuantizer(prefix, weight, config)


class IntQuantLinear(nn.Linear):
    """带量化的线性层，支持权重量化训练"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prefix: str = "",
        config: QuantConfig | None = None,
    ):
        """初始化量化线性层

        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
            prefix: 量化器前缀标识符
            config: 量化配置对象
        """
        # 1. 正常初始化 nn.Linear 的参数（weight, bias 等）
        super().__init__(in_features, out_features, bias=bias)

        # 2. 记录自己的额外信息（不会影响 state_dict 的 key 结构）
        self.use_weight_quant = False  # 是否启用量化
        self.config = config

        # 3. 构建量化器（注意：这里用的是 self.weight）
        self.weight_quantizer = None
        if config is not None:
            self.weight_quantizer = build_weight_quantizer(
                prefix=prefix,
                weight=self.weight,
                config=config,
            )
            # 为 GradualQuantizer 设置 weight 提供者
            if hasattr(self.weight_quantizer, 'set_weight_provider'):
                self.weight_quantizer.set_weight_provider(lambda: self.weight)
            # Keep a copy of full numel before sharding (FSDP/DTensor may change local numel later).
            if hasattr(self.weight_quantizer, "_num_elements"):
                self.weight_quantizer._num_elements_full = self.weight.numel()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播：根据配置决定是否进行假量化

        Args:
            input: 输入张量

        Returns:
            输出张量
        """
        if self.use_weight_quant and self.weight_quantizer is not None:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)

    @classmethod
    def from_float(
        cls,
        prefix: str,
        org_module: nn.Linear,
        config: QuantConfig,
    ) -> "IntQuantLinear":
        """从普通 nn.Linear 创建量化线性层

        Args:
            prefix: 量化器前缀标识符
            org_module: 原始线性层
            config: 量化配置对象

        Returns:
            IntQuantLinear: 新的量化线性层
        """
        # 1. 用 org_module 的形状信息正常创建一层新的 IntQuantLinear
        qlinear = cls(
            in_features=org_module.in_features,
            out_features=org_module.out_features,
            bias=org_module.bias is not None,
            prefix=prefix,
            config=config,
        ).to(org_module.weight.device, dtype=org_module.weight.dtype)
        qlinear.train(org_module.training)

        # 2. 拷贝权重（注意是 copy_，不是复用同一个 Parameter）
        qlinear.weight.data.copy_(org_module.weight.data)
        if org_module.bias is not None and qlinear.bias is not None:
            qlinear.bias.data.copy_(org_module.bias.data)

        return qlinear


def convert_linear(module: nn.Module, prefix: str, config: QuantConfig) -> None:
    """递归替换模块中的所有 nn.Linear 为 IntQuantLinear

    Args:
        module: PyTorch 模块
        prefix: 前缀
        config: 量化配置
    """
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and not isinstance(child, IntQuantLinear):
            setattr(module, name, IntQuantLinear.from_float(child_prefix, child, config))
        else:
            convert_linear(child, child_prefix, config)


def set_weight_parameters(model: nn.Module, requires_grad: bool) -> List[nn.Parameter]:
    """设置权重参数（排除 scale/zero_point）的 requires_grad

    Args:
        model: PyTorch 模型
        requires_grad: 是否需要梯度

    Returns:
        参数列表
    """
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            m.requires_grad = requires_grad
    return params


def weight_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取权重参数（排除 scale/zero_point）

    Args:
        model: PyTorch 模型

    Returns:
        权重参数列表
    """
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            params.append(m)
    return params


def set_quant_parameters(model: nn.Module, requires_grad: bool) -> List[nn.Parameter]:
    """设置量化参数（scale/zero_point）的 requires_grad

    Args:
        model: PyTorch 模型
        requires_grad: 是否需要梯度

    Returns:
        参数列表
    """
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            m.requires_grad = requires_grad
    return params


def quant_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取量化参数（scale/zero_point）

    Args:
        model: PyTorch 模型

    Returns:
        量化参数列表
    """
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            params.append(m)
    return params


def quantizer_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取所有量化器的参数（scale、zero_point 等）

    Args:
        model: PyTorch 模型

    Returns:
        量化器参数列表
    """
    params = []
    for m in model.modules():
        if isinstance(m, IntQuantLinear) and m.weight_quantizer is not None:
            params.extend(list(m.weight_quantizer.parameters()))
    return params


@torch.no_grad()
def reinit_quant_params(model: nn.Module) -> None:
    """重新初始化量化参数（scale/zero_point）

    Args:
        model: PyTorch 模型
    """
    for m in model.modules():
        if isinstance(m, IntQuantLinear) and m.weight_quantizer is not None:
            q = m.weight_quantizer
            scale, zp = q.init_with_weight(m.weight, q.n_bits, q.group_size, clamp_method=q.clamp_method)
            if scale is None or zp is None:
                continue
            scale = scale.to(device=m.weight.device, dtype=m.weight.dtype)
            zp = zp.to(device=m.weight.device, dtype=m.weight.dtype)
            if hasattr(q, "scale") and isinstance(q.scale, nn.Parameter):
                q.scale.data.copy_(scale)
            else:
                q.scale = nn.Parameter(scale)
            if hasattr(q, "zero_point") and isinstance(q.zero_point, nn.Parameter):
                q.zero_point.data.copy_(zp)
            else:
                q.zero_point = nn.Parameter(zp)


@torch.no_grad()
def sanitize_quant_params(model: nn.Module) -> int:
    """清理量化参数（处理 NaN/Inf 值）

    Args:
        model: PyTorch 模型

    Returns:
        int: 修复的参数数量
    """
    repaired = 0
    for m in model.modules():
        if isinstance(m, IntQuantLinear) and m.weight_quantizer is not None:
            q = m.weight_quantizer
            if hasattr(q, "scale") and isinstance(q.scale, nn.Parameter):
                scale = q.scale.data
                if not torch.isfinite(scale).all():
                    repaired += 1
                    scale = torch.nan_to_num(scale, nan=1e-4, posinf=1e4, neginf=1e-4)
                scale.clamp_(1e-4, 1e4)
                q.scale.data.copy_(scale)
            if hasattr(q, "zero_point") and isinstance(q.zero_point, nn.Parameter):
                zp = q.zero_point.data
                if not torch.isfinite(zp).all():
                    repaired += 1
                    zp = torch.nan_to_num(zp, nan=0.0, posinf=float(q.qmax), neginf=float(q.qmin))
                zp.clamp_(q.qmin, q.qmax)
                q.zero_point.data.copy_(zp)
    return repaired


def set_quant_state(model: nn.Module, weight_quant: bool = False) -> None:
    """设置所有量化线性层的量化开关

    Args:
        model: PyTorch 模型
        weight_quant: 是否启用量化
    """
    for m in model.modules():
        if isinstance(m, IntQuantLinear):
            m.use_weight_quant = weight_quant


@torch.no_grad()
def quant_inplace(model: nn.Module) -> None:
    """原位置量化：将量化结果直接写入权重

    Args:
        model: PyTorch 模型
    """
    for _, m in model.named_modules():
        if isinstance(m, IntQuantLinear):
            m.weight.data = m.weight_quantizer(m.weight.data)


def set_op_by_name(layer: nn.Module, name: str, new_module: nn.Module) -> None:
    """按名称设置模块

    Args:
        layer: 父模块
        name: 模块名称（可以包含 . 分隔的层级）
        new_module: 新模块
    """
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)
