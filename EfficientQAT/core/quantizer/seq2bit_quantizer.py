"""2-bit序列量化器，使用独立接口并兼容训练主链路。"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base_quantizer import BaseQuantizer
from .config import QuantConfig
from .ops import clamp_ste, round_ste


class Seq2BitQuantizer(BaseQuantizer):
    """独立2-bit量化器。

    量化级别固定为 {-0.75, -0.25, 0.25, 0.75} * alpha。
    同时暴露 `scale/zero_point/cal_qparams/init_with_weight` 接口以兼容训练与导出流程。
    其中可训练参数仅 `alpha`，`scale/zero_point` 为派生量。
    """

    def __init__(
        self,
        prefix: str,
        weight: torch.Tensor,
        config: QuantConfig,
        group_size: Optional[int] = None,
        enable: bool = True,
    ):
        super().__init__(config=config, group_size=group_size, enable=enable, clamp_method=config.clamp_method)
        self.prefix = prefix
        self.enable = bool(enable)
        self.n_bits = 2
        self._set_qrange(self.n_bits)
        self.eps = 1e-6

        alpha, zero_point = self.init_with_weight(
            weight=weight,
            n_bits=self.n_bits,
            group_size=self.group_size,
            clamp_method=self.clamp_method,
            symmetric=False,
        )
        self.alpha = nn.Parameter(alpha)
        # Keep zero_point as non-trainable parameter so meta-init/FSDP2 weight loader
        # handles it through parameter dispatch path (buffers on meta can fail to copy).
        # self.zero_point = nn.Parameter(zero_point, requires_grad=False)

    @property
    def scale(self) -> torch.Tensor:
        # Compatibility alias for training/export code paths expecting q.scale.
        return self.alpha

    @staticmethod
    def init_with_weight(
        weight: torch.Tensor,
        n_bits: int,
        group_size: int,
        clamp_method: str = "STE",
        symmetric: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if weight is None:
            return None, None
        if group_size is None:
            raise ValueError("group_size must not be None")
        if weight.ndim != 2:
            raise ValueError(f"Seq2BitQuantizer expects 2D weight, got shape={tuple(weight.shape)}")
        if weight.shape[1] % group_size != 0:
            raise ValueError(
                f"in_features must be divisible by group_size, got in={weight.shape[1]}, group={group_size}"
            )
        x = weight.reshape(-1, group_size)
        alpha = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        # zero_point = torch.zeros_like(alpha)
        return alpha, None

    def cal_qparams(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scale = clamp_ste(scale.abs(), 1e-6, 1e4)
        return scale, None

    def _quantize(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        round_zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.reshape(-1, self.group_size)
        s = scale.reshape(-1, 1).clamp_min(self.eps)
        xn = (x / s).clamp(-1.0, 1.0)
        code = round_ste((xn + 0.75) / 0.5).clamp(self.qmin, self.qmax)
        return code

    def _dequantize(
        self,
        x_int: torch.Tensor,
        scale: torch.Tensor,
        round_zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s = scale.reshape(-1, 1).clamp_min(self.eps)
        levels = x_int * 0.5 - 0.75
        return levels * s

    def fake_quant(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = self.alpha if scale is None else scale
        scale, _ = self.cal_qparams(scale, zero_point)
        # CUDA path: prefer seq2bit kernel by default. Fallback to torch path only
        # when kernel constraints are not met.
        if x.is_cuda:
            x_kernel = x.contiguous()
            scale_kernel = scale.reshape(-1).contiguous()
            use_kernel = (
                self.group_size in (64, 128, 256)
                and x_kernel.numel() % self.group_size == 0
                and scale_kernel.numel() == x_kernel.numel() // self.group_size
            )
            if use_kernel:
                from EfficientQAT.core.quantizer.kernel.fake_quant import fake_quant_ste_seq2bit

                return fake_quant_ste_seq2bit(x_kernel, scale_kernel, self.group_size)

        # Torch fallback (CPU / unsupported shape/group).
        if (
            x.numel() % self.group_size == 0
            and scale.numel() == x.numel() // self.group_size
        ):
            ori_shape = x.shape
            xg = x.reshape(-1, self.group_size)
            q = self._quantize(xg, scale, None)
            x_dequant = self._dequantize(q, scale, None)
            return x_dequant.reshape(ori_shape)

        # Last-resort fallback for mismatched runtime shape.
        # Keep behavior explicit to avoid silent wrong reshape.
        if x.shape[-1] % self.group_size != 0:
            raise ValueError(
                f"Seq2BitQuantizer fake_quant expects last dim divisible by group_size, "
                f"got x.shape={tuple(x.shape)}, group_size={self.group_size}"
            )
        ori_shape = x.shape
        xg = x.reshape(-1, self.group_size)
        q = self._quantize(xg, scale, None)
        x_dequant = self._dequantize(q, scale, None)
        return x_dequant.reshape(ori_shape)

    @torch.no_grad()
    def quantize_codes(self, x: torch.Tensor) -> torch.Tensor:
        scale, _ = self.cal_qparams(self.alpha, None)
        q = self._quantize(x, scale, None)
        return q.to(torch.uint8)

    @torch.no_grad()
    def update_qparams_from_weight(self, weight: torch.Tensor) -> None:
        """Re-init qparams from weight; keeps trainable parameter as alpha only."""
        alpha, zp = self.init_with_weight(
            weight,
            self.n_bits,
            self.group_size,
            clamp_method=self.clamp_method,
            symmetric=False,
        )
        alpha = alpha.to(device=weight.device, dtype=weight.dtype)
        self.alpha.data.copy_(alpha)

    @torch.no_grad()
    def sanitize_qparams(self) -> int:
        repaired = 0
        alpha = self.alpha.data
        if not torch.isfinite(alpha).all():
            repaired += 1
            alpha = torch.nan_to_num(alpha, nan=1e-4, posinf=1e4, neginf=1e-4)
        alpha.clamp_(1e-4, 1e4)
        self.alpha.data.copy_(alpha)
        return repaired
