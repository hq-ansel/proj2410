# gradual.py
import torch
from torch import nn

from .uniform_affine import UniformAffineQuantizer
from .config import QuantConfig

class GradualMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.quantization_position_ratio = 0.0
        self.interpolate_ratio = 0.0
    def update_position_ratio(self, new_ratio: float):
        self.quantization_position_ratio = new_ratio
    def update_interpolate_ratio(self, new_ratio: float):
        self.interpolate_ratio = new_ratio
    def _split_quant_groups(self, x):
        """
        根据量化位置比例计算应被量化的组数
        
        Args:
            x (torch.Tensor): 输入张量，其第一维度表示总组数
        
        Returns:
            int: 需要被量化的组数，至少为1
        """
        total_groups = x.shape[0]
        quantized_groups = max(int(total_groups * self.quantization_position_ratio), 1)
        return quantized_groups

class GradualQuantizer(GradualMixin,UniformAffineQuantizer):
    def __init__(self, prefix: str, weight: torch.Tensor, config:QuantConfig):
        super().__init__(prefix ,weight, config)
        self.quantization_position_ratio = 0.0
        self.interpolate_ratio = 0.0

    def fake_quant(self, x):
        scale, round_zero_point = self.cal_qparams(
            self.scale,
            self.zero_point)
        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)

        quantized_groups = self._split_quant_groups(x)
        if quantized_groups >= x.shape[0]:
            # all quantized
            x_int = self._quantize(x, scale, round_zero_point)
            if self.is_tracking:
                x_int = self.weight_freeze_tracker(x_int)
            x_dequant = self._dequantize(x_int, scale, round_zero_point)
            return x_dequant.reshape(ori_shape)
        elif quantized_groups <= 0:
            # none quantized
            return x.reshape(ori_shape)
        else:
            # partial quantized
            x_quant = x[:quantized_groups]
            x_float = x[quantized_groups:]

            x_int = self._quantize(x_quant, scale, round_zero_point)
            if self.is_tracking:
                x_int = self.weight_freeze_tracker(x_int)
            x_dequant = self._dequantize(x_int, scale, round_zero_point)

            if self.interpolate_ratio > 0.0:
                interp_ratio = self.interpolate_ratio
                x_dequant = x_dequant * interp_ratio + x_quant * (1 - interp_ratio)

            x_out = torch.cat([x_dequant, x_float], dim=0)
            return x_out.reshape(ori_shape)