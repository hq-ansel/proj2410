# uniform_affine.py
from dataclasses import dataclass
import torch
from torch import nn

from .base_quantizer import BaseQuantizer
from .tracking import TrackOscillation
from .config import QuantConfig


@dataclass
class QuantLog:
    prefix: str
    amax_diff: float
    mean_diff: float


class UniformAffineQuantizer(BaseQuantizer):
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

    def fake_quant(self, x):
        scale, round_zero_point = self.cal_qparams(
            self.scale,
            self.zero_point)
        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)
        # freezing weights
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
