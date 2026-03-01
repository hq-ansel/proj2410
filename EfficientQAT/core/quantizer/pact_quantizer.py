"""PACT (Parametric Clipping Activation) 量化器实现

PACT 通过引入可学习的激活截断参数 alpha 来控制激活范围，
然后在 [-alpha, alpha] 范围内进行均匀量化。

参考: "PACT: Parameterized Clipping Activation for Quantized Neural Networks"
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_quantizer import BaseQuantizer
from .config import QuantConfig
from .ops import clamp_ste, round_ste


def _to_local_if_dtensor(x: torch.Tensor) -> torch.Tensor:
    """Return local shard for DTensor-like inputs; no-op for regular Tensor."""
    if hasattr(x, "to_local"):
        return x.to_local()
    return x


class PACTActivationQuantizer(BaseQuantizer):
    """PACT 激活量化器。

    PACT 使用可学习的截断参数 alpha 来限制激活范围:
    - 前向传播：先截断激活值到 [-alpha, alpha]，然后进行均匀量化
    - 反向传播：alpha 通过 STE 学习最优截断范围

    量化公式：
        x_clipped = clamp(x, -alpha, alpha)
        x_quant = round(x_clipped / scale) * scale
    其中 scale = 2 * alpha / (2^n_bits - 1)
    """

    def __init__(
        self,
        prefix: str,
        activation_shape: torch.Size,
        config: QuantConfig,
        group_size: Optional[int] = None,
        enable: bool = True,
    ):
        super().__init__(config=config, group_size=group_size, enable=enable, clamp_method=config.clamp_method)
        self.prefix = prefix
        self.enable = bool(enable)
        self.n_bits = config.activation_n_bits if hasattr(config, 'activation_n_bits') else config.n_bits
        self._set_qrange(self.n_bits)
        self.eps = 1e-6

        # 初始化 alpha 为激活的最大绝对值
        # 对于激活量化，我们使用 per-tensor 或 per-channel 的 alpha
        init_alpha = self._init_alpha(activation_shape)
        self.alpha = nn.Parameter(init_alpha, requires_grad=True)

    def _init_alpha(self, activation_shape: torch.Size) -> torch.Tensor:
        """初始化 alpha 参数。

        对于 per-tensor 量化，alpha 是标量
        对于 per-channel 量化，alpha 是每个通道一个值
        """
        if self.group_size is None or self.group_size == -1:
            # Per-tensor: single alpha
            return torch.ones(1, dtype=torch.float32) * 0.5
        else:
            # Per-channel: alpha for each output channel
            # 假设 activation_shape 是 (batch, seq_len, out_features)
            num_channels = activation_shape[-1]
            return torch.ones(num_channels, dtype=torch.float32) * 0.5

    @staticmethod
    def init_alpha_from_data(
        data: torch.Tensor,
        n_bits: int,
        group_size: int,
        percentile: float = 99.99,
    ) -> torch.Tensor:
        """从数据中初始化 alpha。

        使用 percentile 方法设置 alpha 为数据绝对值的某个分位数。
        """
        data = _to_local_if_dtensor(data)
        if group_size is None or group_size == -1:
            # Per-tensor
            alpha = torch.tensor([data.abs().quantile(percentile / 100.0)], dtype=torch.float32)
        else:
            # Per-channel
            if data.ndim == 2:
                # (batch * seq_len, out_features)
                alpha = data.abs().quantile(percentile / 100.0, dim=0, keepdim=True).squeeze(0)
            else:
                # Reshape to (N, out_features)
                data_flat = data.reshape(-1, data.shape[-1])
                alpha = data_flat.abs().quantile(percentile / 100.0, dim=0)
        return alpha.clamp_min(1e-6)

    @property
    def scale(self) -> torch.Tensor:
        """计算量化 scale: scale = 2 * alpha / (2^n_bits - 1)"""
        return 2 * self.alpha / (2 ** self.n_bits - 1)

    @property
    def zero_point(self) -> Optional[torch.Tensor]:
        """对称量化，zero_point 为 None 或 0"""
        return None

    def cal_qparams(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """返回量化参数 (scale, zero_point)。"""
        return self.scale, self.zero_point

    def _quantize(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """执行量化操作。"""
        # 量化到整数级别
        x_scaled = x / scale
        if zero_point is not None:
            x_scaled = x_scaled + zero_point
        x_int = round_ste(x_scaled)
        # 限制量化范围
        qmin, qmax = self.qmin, self.qmax
        x_int = x_int.clamp(qmin, qmax)
        # 反量化回浮点
        x_dequant = (x_int - zero_point) * scale if zero_point is not None else x_int * scale
        return x_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PACT 前向传播：先截断，再量化。

        Args:
            x: 输入激活张量

        Returns:
            量化后的激活张量
        """
        if not self.enable:
            return x

        # PACT 截断
        x_clipped = x.clamp(-self.alpha, self.alpha)

        # 量化
        scale, zero_point = self.cal_qparams(self.scale, self.zero_point)

        # 对于 per-channel 量化，需要调整 scale 的形状
        if scale.ndim == 1 and x_clipped.ndim == 3:
            # scale: (out_features,) -> (1, 1, out_features)
            scale = scale.view(1, 1, -1)
        elif scale.ndim == 1 and x_clipped.ndim == 2:
            # scale: (out_features,) -> (1, out_features)
            scale = scale.view(1, -1)

        x_quant = self._quantize(x_clipped, scale, zero_point)

        # 保留梯度（x_clipped 已经有梯度，不需要额外处理）
        return x_quant

    def get_alpha_grad_stats(self) -> dict:
        """获取 alpha 梯度的统计信息。"""
        if self.alpha.grad is None:
            return {"norm": 0.0, "mean": 0.0, "max": 0.0}
        grad = self.alpha.grad
        return {
            "norm": float(grad.norm().item()),
            "mean": float(grad.mean().item()),
            "max": float(grad.abs().max().item()),
        }


class PACTQuantLinear(nn.Linear):
    """带 PACT 激活量化的线性层。

    特点：
    - 使用权重量化（通过 IntQuantLinear 的机制）
    - 使用 PACT 激活量化
    - alpha 参数可学习
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prefix: str = "",
        config: QuantConfig | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.prefix = prefix
        self.use_weight_quant = False
        self.use_activation_quant = False
        self.config = config

        # 权重量化器
        self.weight_quantizer = None
        if config is not None:
            from ..linear.int_quant_linear import build_weight_quantizer
            self.weight_quantizer = build_weight_quantizer(prefix, self.weight, config)
            if hasattr(self.weight_quantizer, 'set_weight_provider'):
                self.weight_quantizer.set_weight_provider(lambda: self.weight)

        # PACT 激活量化器（在 forward 时延迟初始化，因为我们不知道激活形状）
        self.activation_quantizer: Optional[PACTActivationQuantizer] = None
        self._activation_shape: Optional[torch.Size] = None

    def _init_activation_quantizer(self, activation_shape: torch.Size):
        """延迟初始化激活量化器。"""
        if self.activation_quantizer is None and self.config is not None:
            self.activation_quantizer = PACTActivationQuantizer(
                prefix=f"{self.prefix}_act",
                activation_shape=activation_shape,
                config=self.config,
                group_size=getattr(self.config, 'activation_group_size', None),
                enable=True,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播：权重量化 + PACT 激活量化。"""
        # 权重量化
        if self.use_weight_quant and self.weight_quantizer is not None:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        # 线性变换
        output = F.linear(input, weight, self.bias)

        # PACT 激活量化
        if self.use_activation_quant:
            if self.activation_quantizer is None:
                self._init_activation_quantizer(output.shape)
            if self.activation_quantizer is not None:
                output = self.activation_quantizer(output)

        return output

    def set_activation_quant_state(self, enabled: bool):
        """设置激活量化的启用状态。"""
        self.use_activation_quant = enabled
        if enabled and self.activation_quantizer is not None:
            self.activation_quantizer.enable = True


def pact_forward(x: torch.Tensor, alpha: torch.Tensor, n_bits: int, group_size: int = -1) -> torch.Tensor:
    """PACT 前向传播的函数式实现。

    Args:
        x: 输入激活
        alpha: 可学习的截断参数
        n_bits: 量化位数
        group_size: 量化组大小，-1 表示 per-tensor

    Returns:
        量化后的输出
    """
    # PACT 截断
    x_clipped = x.clamp(-alpha, alpha)

    # 计算 scale
    scale = 2 * alpha / (2 ** n_bits - 1)

    # 量化
    if group_size == -1:
        # Per-tensor
        x_scaled = x_clipped / scale
        x_int = round_ste(x_scaled)
        x_int = x_int.clamp(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
        x_quant = x_int * scale
    else:
        # Per-channel
        if scale.ndim == 1:
            scale = scale.view(1, 1, -1) if x_clipped.ndim == 3 else scale.view(1, -1)
        x_scaled = x_clipped / scale
        x_int = round_ste(x_scaled)
        qmin, qmax = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
        x_int = x_int.clamp(qmin, qmax)
        x_quant = x_int * scale

    return x_quant


class PACTAlphaGradHook:
    """用于收集 alpha 梯度的 hook 类。"""

    def __init__(self):
        self.grad_stats = {"norm": 0.0, "count": 0}

    def __call__(self, grad: torch.Tensor) -> None:
        """更新梯度统计。"""
        self.grad_stats["norm"] += float(grad.norm().item())
        self.grad_stats["count"] += 1

    def reset(self):
        """重置统计。"""
        self.grad_stats = {"norm": 0.0, "count": 0}

    @property
    def avg_norm(self) -> float:
        """返回平均梯度范数。"""
        if self.grad_stats["count"] == 0:
            return 0.0
        return self.grad_stats["norm"] / self.grad_stats["count"]


__all__ = [
    "PACTActivationQuantizer",
    "PACTQuantLinear",
    "pact_forward",
    "PACTAlphaGradHook",
]
