# base_quantizer.py
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .ops import round_ste, clamp_ste, clamp_mad
from .config import QuantConfig

class BaseQuantizer(nn.Module):
    def __init__(self,
                config: QuantConfig,
                n_bits: int = 8,
                group_size: Optional[int] = None,
                enable: bool = True,
                clamp_method: str = "STE",
                **kwargs):
        super().__init__()
        # 使用默认配置或用户提供的配置
        config = config or QuantConfig()
        
        # 使用直接参数覆盖配置对象中的值（如果提供了）
        self.n_bits = n_bits if n_bits is not None else config.n_bits
        self.qmin = 0
        self.qmax = (1 << self.n_bits) - 1
        self.group_size = group_size if group_size is not None else config.group_size
        self.enable = enable if enable is not None else config.enable
        self.clamp_method = clamp_method if clamp_method is not None else config.clamp_method
        
        assert 2 <= self.n_bits <= 16
        if self.group_size is None:
            raise ValueError("group_size must not be None")

    def change_n_bits(self, n_bits: int) -> None:
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = (1 << n_bits) - 1

    @staticmethod
    def init_with_weight(weight: torch.Tensor,
                        n_bits: int,
                        group_size: int,
                        clamp_method: str = "STE"
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if weight is None:
            print("weight is None")
            return None, None
        if group_size is None:
            raise ValueError("group_size must not be None")
        with torch.no_grad():
            x = weight.reshape(-1,group_size)
            xmin = x.amin([-1], keepdim=True)
            xmax =  x.amax([-1], keepdim=True)
            x_range = xmax - xmin
            scale = x_range / (2**n_bits-1)
            if clamp_method == "STE":
                scale = clamp_ste(scale, 1e-4, 1e4)
            elif clamp_method == "MAD":
                scale = clamp_mad(scale, 1e-4, 1e4)
            zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
            return scale, zero_point.round()

    def cal_qparams(self,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, round_zero_point = None,None
        if self.clamp_method == "STE":
            scale_dtype = scale.dtype
            scale = clamp_ste(scale,1e-4, 1e4).to(scale_dtype)
            round_zero_point = clamp_ste(round_ste(zero_point), self.qmin, self.qmax)
        elif self.clamp_method == "MAD":
            scale = clamp_mad(scale, 1e-4, 1e4)
            round_zero_point = clamp_mad(round_ste(zero_point), self.qmin, self.qmax)
        return scale, round_zero_point

    def _quantize(self,
                  x: torch.Tensor,
                  scale: torch.Tensor,
                  round_zero_point: Optional[torch.Tensor]
                  ) -> torch.Tensor:
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        return x_int
    
    def _dequantize(self,
                    x_int: torch.Tensor,
                    scale: torch.Tensor,
                    round_zero_point: Optional[torch.Tensor]
                    ) -> torch.Tensor:
        if round_zero_point is not None:
            x_int = x_int.sub(round_zero_point)
        x_float = x_int.mul(scale)
        return x_float
    
    def fake_quant(self,
                   x: torch.Tensor
                   ) -> torch.Tensor:
        scale, round_zero_point = self.cal_qparams(self.scale,
                                                   self.zero_point)
        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = self._quantize(x, scale, round_zero_point)
        x_dequant = self._dequantize(x_int, scale, round_zero_point)
        return x_dequant.reshape(ori_shape)
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        if self.n_bits >= 16 or not self.enable:
            return x
        return self.fake_quant(x)