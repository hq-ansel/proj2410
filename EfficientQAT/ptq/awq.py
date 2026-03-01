"""
AWQ (Activation-aware Weight Quantization) PTQ 实现

AWQ 是一种后训练量化方法，通过激活值的重要性来保护显著权重。
核心思想：
1. 使用校准数据计算每个权重的激活重要性（activation magnitude）
2. 根据重要性对权重进行缩放，保护显著权重
3. 对缩放后的权重进行 per-channel/per-group 量化

参考：https://arxiv.org/abs/2306.00978
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class AWQConfig:
    """AWQ 量化配置

    Attributes:
        n_bits: 量化位数 (默认：4)
        group_size: 分组大小 (默认：128)
        zero_point: 是否使用零点 (默认：True)
        calibration_samples: 校准样本数量 (默认：128)
        auto_scale: 是否自动缩放 (默认：True)
        auto_clip: 是否自动截断 (默认：True)
        search_scale: 是否搜索缩放因子 (默认：False，简化实现)
        search_steps: 搜索步数 (默认：20)
    """
    n_bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    calibration_samples: int = 128
    auto_scale: bool = True
    auto_clip: bool = True
    search_scale: bool = False
    search_steps: int = 20


class AWQQuantizer:
    """AWQ 量化器

    对 Linear 层进行 AWQ 量化，支持 per-group 量化和激活感知缩放。
    """

    def __init__(
        self,
        module: nn.Linear,
        config: AWQConfig,
        name: str = "",
    ):
        self.module = module
        self.config = config
        self.name = name

        # 量化参数
        self.n_bits = config.n_bits
        self.group_size = config.group_size
        self.zero_point = config.zero_point

        # 量化后的权重和参数
        self.qweight = None  # 量化权重
        self.scales = None   # 缩放因子 (float16)
        self.qzeros = None   # 零点 (int32)

        # 激活感知缩放因子
        self.act_scales = None

        # 重要性缩放因子
        self.importance_scale = None

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight

    @property
    def in_features(self) -> int:
        return self.module.in_features

    @property
    def out_features(self) -> int:
        return self.module.out_features

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def calculate_activation_scales(
        self,
        dataloader: Any,
        num_samples: int = 128,
    ) -> torch.Tensor:
        """计算输入激活的缩放因子（用于重要性估计）

        通过校准数据计算每个输入通道的激活绝对值平均值。
        """
        self.module.eval()
        act_scales = None
        samples_processed = 0

        for batch in dataloader:
            if samples_processed >= num_samples:
                break

            # 获取输入
            if isinstance(batch, dict):
                inputs = batch.get('input_ids', batch)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            if isinstance(inputs, torch.Tensor):
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(0)

                # 只处理输入的前 in_features 个通道
                if inputs.shape[-1] != self.in_features:
                    continue

                with torch.no_grad():
                    self.module(inputs.to(self.device))

                # 计算每个输入通道的绝对值均值
                inp_flat = inputs.reshape(-1, inputs.shape[-1])
                scale = inp_flat.abs().mean(dim=0)  # [in_features]

                if act_scales is None:
                    act_scales = torch.zeros_like(scale)
                act_scales += scale

                samples_processed += inputs.shape[0]

        if act_scales is None:
            # 默认均匀重要性
            act_scales = torch.ones(self.in_features, device=self.device)
        else:
            act_scales = act_scales / max(1, samples_processed)

        return act_scales

    def calculate_importance(
        self,
        dataloader: Any,
        num_samples: int = 128,
    ) -> torch.Tensor:
        """计算权重的重要性

        重要性 = |W| * 激活重要性
        其中激活重要性通过校准数据的激活绝对值均值估计。
        """
        # 获取激活重要性 [in_features]
        act_scales = self.calculate_activation_scales(dataloader, num_samples)

        # 计算权重重要性
        # weight: [out_features, in_features]
        weight = self.weight.abs()

        # 重要性 = |W| * act_scales (broadcast 到 in_features 维度)
        importance = weight * act_scales.unsqueeze(0)

        return importance

    def _quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """权重量化 (per-group)"""
        assert self.group_size > 0, "group_size must be positive"

        # 重塑为 [out_features, in_features]
        orig_shape = weight.shape
        weight_flat = weight.reshape(-1)

        # 计算需要填充的大小
        num_groups = (weight_flat.shape[0] + self.group_size - 1) // self.group_size
        padded_size = num_groups * self.group_size

        if padded_size > weight_flat.shape[0]:
            weight_flat = F.pad(weight_flat, (0, padded_size - weight_flat.shape[0]))

        # 重塑为 [num_groups, group_size]
        weight_groups = weight_flat.reshape(num_groups, self.group_size)

        # 量化参数
        qmin = 0
        qmax = 2 ** self.n_bits - 1

        # 计算每组的 min/max
        weight_min = weight_groups.min(dim=-1, keepdim=True)[0]  # [num_groups, 1]
        weight_max = weight_groups.max(dim=-1, keepdim=True)[0]  # [num_groups, 1]

        if self.zero_point:
            # 非对称量化
            scale = (weight_max - weight_min) / (qmax - qmin)
            zero_point = (qmin - weight_min / scale).round().clamp(qmin, qmax).to(torch.int32)
        else:
            # 对称量化
            weight_max_abs = weight_groups.abs().max(dim=-1, keepdim=True)[0]
            scale = weight_max_abs / (qmax // 2)
            zero_point = torch.zeros_like(scale)

        scale = scale.clamp(min=1e-8)

        # 量化
        q_weight = ((weight_groups / scale) + zero_point).round().clamp(qmin, qmax).to(torch.int32)

        # 存储 scales 和 qzeros [num_groups]
        self.scales = scale.reshape(num_groups)
        self.qzeros = zero_point.reshape(num_groups) if self.zero_point else None

        # 返回原始形状
        return q_weight.reshape(orig_shape)

    def _dequantize_weight(self, q_weight: torch.Tensor) -> torch.Tensor:
        """权重反量化"""
        if self.scales is None:
            raise ValueError("Scales not computed yet")

        orig_shape = q_weight.shape
        q_weight_flat = q_weight.reshape(-1).float()

        # 重塑为 [num_groups, group_size]
        num_groups = q_weight_flat.shape[0] // self.group_size
        q_weight_groups = q_weight_flat.reshape(num_groups, self.group_size)

        # 反量化
        if self.zero_point and self.qzeros is not None:
            qzeros_flat = self.qzeros.reshape(num_groups, 1)
            scales_flat = self.scales.reshape(num_groups, 1)
            dequant = (q_weight_groups - qzeros_flat) * scales_flat
        else:
            scales_flat = self.scales.reshape(num_groups, 1)
            dequant = q_weight_groups * scales_flat

        return dequant.reshape(orig_shape)

    def quantize(
        self,
        dataloader: Any = None,
        num_samples: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """执行 AWQ 量化

        Args:
            dataloader: 校准数据加载器
            num_samples: 校准样本数量

        Returns:
            (qweight, scales, qzeros) 元组
        """
        self.module.eval()

        with torch.no_grad():
            if dataloader is not None and self.config.search_scale:
                # 1. 计算重要性
                importance = self.calculate_importance(dataloader, num_samples)

                # 2. 计算缩放因子 (简化版本：使用重要性作为缩放)
                # AWQ 的核心：根据激活重要性缩放权重
                act_scales = importance / (self.weight.abs() + 1e-8)
                self.act_scales = act_scales.clamp(min=1e-5)

                # 3. 应用缩放因子
                scaled_weight = self.weight.data * self.act_scales
            elif dataloader is not None:
                # 简化版本：只使用激活感知，不搜索缩放
                act_scales = self.calculate_activation_scales(dataloader, num_samples)
                self.act_scales = act_scales.clamp(min=1e-5)

                # 使用激活重要性缩放权重
                scaled_weight = self.weight.data * act_scales.unsqueeze(0)
            else:
                # 没有校准数据，直接量化
                scaled_weight = self.weight.data

            # 4. 量化权重
            q_weight = self._quantize_weight(scaled_weight)
            self.qweight = q_weight

        return self.qweight, self.scales, self.qzeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用前向传播（使用解包后的权重进行矩阵乘法）"""
        if self.qweight is None:
            raise ValueError("Must call quantize() first")

        # 反量化权重
        weight = self._dequantize_weight(self.qweight)

        # 如果有缩放因子，需要反缩放
        if self.act_scales is not None:
            weight = weight / self.act_scales.unsqueeze(0)

        # 重塑为 [out_features, in_features]
        weight = weight.reshape(self.out_features, self.in_features)

        return F.linear(x, weight, self.module.bias)


def apply_awq_to_model(
    model: nn.Module,
    config: AWQConfig,
    calib_dataloader: Any = None,
    verbose: bool = False,
) -> Dict[str, AWQQuantizer]:
    """对模型应用 AWQ 量化

    Args:
        model: 要量化的模型
        config: AWQ 配置
        calib_dataloader: 校准数据加载器
        verbose: 是否打印详细信息

    Returns:
        量化器字典，key 为层名称，value 为 AWQQuantizer
    """
    quantizers = {}

    # 找到所有 Linear 层并创建量化器
    layers_to_quantize = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 跳过输出层（通常不量化）
            if 'lm_head' in name or 'output' in name:
                if verbose:
                    print(f"Skipping {name} (output layer)")
                continue

            if verbose:
                print(f"Applying AWQ to {name}: {module.in_features} -> {module.out_features}")

            # 创建量化器
            quantizer = AWQQuantizer(module, config, name=name)
            quantizers[name] = quantizer
            layers_to_quantize.append((name, module, quantizer))

    # 先执行量化（在替换 forward 之前）
    for name, module, quantizer in layers_to_quantize:
        quantizer.quantize(calib_dataloader, config.calibration_samples)

    # 量化完成后再替换 forward 方法
    for name, module, quantizer in layers_to_quantize:
        # 使用原始权重进行前向传播（量化器保存了量化参数）
        original_forward = module.forward
        def make_quantized_forward(q, orig_fwd):
            def forward(x):
                return q.forward(x)
            return forward
        module.forward = make_quantized_forward(quantizer, original_forward)

    if verbose:
        print(f"Applied AWQ to {len(quantizers)} layers")

    return quantizers


__all__ = [
    "AWQConfig",
    "AWQQuantizer",
    "apply_awq_to_model",
]
