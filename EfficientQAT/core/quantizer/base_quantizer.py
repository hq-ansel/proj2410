# base_quantizer.py
"""量化器基类，提供量化/反量化的核心功能"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .ops import round_ste, clamp_ste, clamp_mad
from .config import QuantConfig

class BaseQuantizer(nn.Module):
    """量化器基类，提供量化/反量化的核心功能"""

    def __init__(self,
                config: Optional[QuantConfig] = None,
                n_bits: int = 8,
                group_size: Optional[int] = None,
                enable: bool = True,
                clamp_method: str = "STE"):
        """初始化量化器

        Args:
            config: 量化配置对象，可选
            n_bits: 量化位数
            group_size: 分组大小
            enable: 是否启用量化
            clamp_method: 截断方法（"STE" 或 "MAD"）
        """
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
        """动态修改量化位数

        Args:
            n_bits: 新的量化位数
        """
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = (1 << n_bits) - 1

    @staticmethod
    def init_with_weight(weight: torch.Tensor,
                        n_bits: int,
                        group_size: int,
                        clamp_method: str = "STE"
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据权重初始化 scale 和 zero_point

        Args:
            weight: 权重张量
            n_bits: 量化位数
            group_size: 分组大小
            clamp_method: 截断方法

        Returns:
            scale: 缩放因子
            zero_point: 零点偏移
        """
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
        """计算量化参数（scale 和 zero_point）并应用截断

        Args:
            scale: 缩放因子
            zero_point: 零点偏移

        Returns:
            scale: 截断后的缩放因子
            round_zero_point: 截断并舍入后的零点
        """
        min_scale = 1e-5
        max_scale = 1e4
        if self.clamp_method == "STE":
            scale_dtype = scale.dtype
            sign = torch.where(scale >= 0, torch.ones_like(scale), -torch.ones_like(scale))
            scale = clamp_ste(scale.abs(), min_scale, max_scale).to(scale_dtype) * sign
            round_zero_point = clamp_ste(round_ste(zero_point), self.qmin, self.qmax)
        elif self.clamp_method == "MAD":
            sign = torch.where(scale >= 0, torch.ones_like(scale), -torch.ones_like(scale))
            scale = clamp_mad(scale.abs(), min_scale, max_scale) * sign
            round_zero_point = clamp_mad(round_ste(zero_point), self.qmin, self.qmax)
        return scale, round_zero_point

    def _quantize(self,
                  x: torch.Tensor,
                  scale: torch.Tensor,
                  round_zero_point: Optional[torch.Tensor]
                  ) -> torch.Tensor:
        """量化输入张量

        Args:
            x: 输入张量
            scale: 缩放因子
            round_zero_point: 舍入后的零点偏移

        Returns:
            量化后的整数张量
        """
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
        """反量化整数张量

        Args:
            x_int: 量化后的整数张量
            scale: 缩放因子
            round_zero_point: 舍入后的零点偏移

        Returns:
            反量化后的浮点张量
        """
        if round_zero_point is not None:
            x_int = x_int.sub(round_zero_point)
        x_float = x_int.mul(scale)
        return x_float

    def fake_quant(self,
                   x: torch.Tensor
                   ) -> torch.Tensor:
        """假量化：量化后反量化，用于可微分量化训练

        Args:
            x: 输入张量

        Returns:
            假量化后的张量
        """
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
        """前向传播：根据配置决定是否进行假量化

        Args:
            x: 输入张量

        Returns:
            假量化后的张量或原张量
        """
        if self.n_bits >= 16 or not self.enable:
            return x
        return self.fake_quant(x)
