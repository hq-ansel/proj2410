# uniform_affine.py
"""均匀仿射量化器实现"""
from dataclasses import dataclass
import torch
from torch import nn

from EfficientQAT.core.quantizer.kernel.fake_quant import fake_quant_ste

from .base_quantizer import BaseQuantizer
from .tracking import TrackOscillation
from .config import QuantConfig


@dataclass
class QuantLog:
    """量化统计日志，记录量化前后的差异"""
    prefix: str  # 权重前缀标识符
    amax_diff: float  # 最大绝对值差异
    mean_diff: float  # 平均绝对值差异


class UniformAffineQuantizer(BaseQuantizer):
    """均匀仿射量化器，所有 group 等优先级量化"""
    def __init__(self,
                 prefix: str,
                 weight: torch.Tensor,
                 config: QuantConfig,
                 *args, **kwargs):
        # 先把 config 传给下一个类
        # 如果 BaseQuantizer 的签名是 __init__(..., config: QuantConfig, ...)
        super().__init__(config=config, *args, **kwargs)
        self.prefix = prefix
        self.enable = True
        self.clamp_method = config.clamp_method
        self.is_tracking = config.is_tracking
        self.stat_quant = config.stat_quant

        scale, zp = BaseQuantizer.init_with_weight(
            weight, self.n_bits, self.group_size, clamp_method=self.clamp_method
        )
        self.scale = nn.Parameter(scale)
        self.zero_point = nn.Parameter(zp)

        if self.is_tracking:
            self.weight_freeze_tracker = TrackOscillation(
                momentum=config.freeze_momentum,
                freeze_threshold=config.freeze_threshold,
                use_ema_x_int=True,
            )
        if self.stat_quant:
            self.quant_stat_log = QuantLog(
                prefix=self.prefix,
                amax_diff=0.0,
                mean_diff=0.0)

    def fake_quant(self,
                   x: torch.Tensor,
                   scale: torch.Tensor = None,
                   zero_point: torch.Tensor = None) -> torch.Tensor:
        """假量化：量化后反量化，用于可微分量化训练

        Args:
            x: 输入张量
            scale: 可选外部 scale（用于对比/梯度注入）
            zero_point: 可选外部 zero_point（用于对比/梯度注入）
        """
        scale = self.scale if scale is None else scale
        zero_point = self.zero_point if zero_point is None else zero_point
        return fake_quant_ste(x, scale, zero_point, self.qmin, self.qmax,self.group_size)
        if zero_point is None:
            # 仅 clamp scale，zero_point 走 None 分支
            scale, _ = self.cal_qparams(scale, torch.zeros_like(scale))
            round_zero_point = None
        else:
            scale, round_zero_point = self.cal_qparams(scale, zero_point)
        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)
        # freezing weights - 冻结权重（如果启用了追踪）
        x_int = self._quantize(x, scale, round_zero_point)
        if self.is_tracking:
            x_int = self.weight_freeze_tracker(x_int)
        x_dequant = self._dequantize(x_int, scale, round_zero_point)
        if self.stat_quant:
            with torch.no_grad():
                amax_diff = (x_dequant - x).abs().amax().item()
                mean_diff = (x_dequant - x).abs().mean().item()
                self.quant_stat_log.amax_diff = amax_diff
                self.quant_stat_log.mean_diff = mean_diff
        return x_dequant.reshape(ori_shape)
