# gradual.py
import torch
from torch import nn
from typing import Iterable, List

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


def _collect_gradual_quantizers(module: nn.Module) -> List[GradualQuantizer]:
    """
    Gather all GradualQuantizer instances from a module tree.
    """
    return [m for m in module.modules() if isinstance(m, GradualQuantizer)]


class GradualQuantContext:
    """
    Context manager that updates quantization_position_ratio for all GradualQuantizer
    modules based on training progress.

    Usage:
    -------
    >>> manager = GradualQuantContext(model, total_steps=1000, warmup_steps=100)
    >>> with manager as sched:
    ...     for step in range(1, 1001):
    ...         sched.step(step)  # sync ratio for this step
    ...         loss = model(input_ids)
    ...         loss.backward()

    The ratio linearly increases from start_ratio to end_ratio after warmup and
    is clamped to [0, 1].
    """

    def __init__(
        self,
        module: nn.Module,
        total_steps: int,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
        warmup_steps: int = 0,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        self.total_steps = total_steps
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_steps = warmup_steps
        self.quantizers: List[GradualQuantizer] = _collect_gradual_quantizers(module)
        self._orig: List[float] = []

    def __enter__(self):
        self._orig = [q.quantization_position_ratio for q in self.quantizers]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore original ratios
        for q, ratio in zip(self.quantizers, self._orig):
            q.update_position_ratio(ratio)

    def _compute_ratio(self, step: int) -> float:
        """
        计算当前步骤的量化比例值
        
        Args:
            step (int): 当前训练步骤数
        
        Returns:
            float: 计算得到的比例值，范围在0.0到1.0之间
        """
        if step <= self.warmup_steps:
            progress = 0.0
        else:
            denom = max(self.total_steps - self.warmup_steps, 1)
            progress = min(max((step - self.warmup_steps) / denom, 0.0), 1.0)
        ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)
        return float(min(max(ratio, 0.0), 1.0))

    def step(self, step: int):
        """
        Update all tracked quantizers for the current global step.
        """
        ratio = self._compute_ratio(step)
        for q in self.quantizers:
            q.update_position_ratio(ratio)
