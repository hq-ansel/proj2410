"""2-bit序列量化器，兼容全量量化与渐进调度。"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base_quantizer import BaseQuantizer
from .config import QuantConfig
from .gradual import GradualMixin
from .ops import clamp_ste, round_ste


def _to_local_if_dtensor(x: torch.Tensor) -> torch.Tensor:
    """Return local shard for DTensor-like inputs; no-op for regular Tensor."""
    if hasattr(x, "to_local"):
        return x.to_local()
    return x


def _align_alpha_numel(alpha: torch.Tensor, target_numel: int) -> torch.Tensor:
    """Align alpha groups to local parameter shard size under uneven DTensor sharding."""
    flat = alpha.reshape(-1)
    cur = flat.numel()
    if cur == target_numel:
        return flat
    if cur > target_numel:
        return flat[:target_numel]
    if cur == 0:
        return torch.full((target_numel,), 1e-4, device=alpha.device, dtype=alpha.dtype)
    pad = flat[-1:].expand(target_numel - cur)
    return torch.cat([flat, pad], dim=0)


class Seq2BitQuantizer(GradualMixin, BaseQuantizer):
    """2-bit weight quantizer with optional gradual scheduling controls."""

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
        self.quantization_position_ratio = 1.0

        self._num_elements = weight.numel()
        self._device = weight.device
        self._get_weight_for_priority = lambda: None
        self._current_step: Optional[int] = None
        self._ramp_start_steps: Dict[int, int] = {}
        self._prev_group_mask: Optional[torch.BoolTensor] = None

        self.ramp_len = max(int(getattr(config, "ramp_len", 0)), 0)
        self.ramp_mode = getattr(config, "ramp_mode", "linear")
        self.ramp_sigmoid_a = float(getattr(config, "ramp_sigmoid_a", 10.0))
        if self.ramp_mode not in ("linear", "sigmoid"):
            self.ramp_mode = "linear"

        # Keep zero_point as non-trainable parameter so meta-init/FSDP2 weight loader
        # handles it through parameter dispatch path (buffers on meta can fail to copy).
        # self.zero_point = nn.Parameter(zero_point, requires_grad=False)
        _ = zero_point

    @property
    def scale(self) -> torch.Tensor:
        return self.alpha

    def set_weight_provider(self, provider_fn) -> None:
        self._get_weight_for_priority = provider_fn

    def get_weight_for_priority(self):
        return self._get_weight_for_priority()

    def set_current_step(self, step: int) -> None:
        self._current_step = int(step)

    def _update_ramp_state(self, mask: torch.BoolTensor) -> None:
        if self.ramp_len <= 0:
            self._prev_group_mask = mask.detach().clone()
            self._ramp_start_steps.clear()
            return
        if self._current_step is None:
            self._prev_group_mask = None
            self._ramp_start_steps.clear()
            return

        if self._prev_group_mask is None or self._prev_group_mask.numel() != mask.numel():
            prev = torch.zeros_like(mask, device=mask.device)
        else:
            prev = self._prev_group_mask.to(device=mask.device)

        new_mask = mask & (~prev)
        if new_mask.any():
            for idx in torch.nonzero(new_mask, as_tuple=True)[0].tolist():
                self._ramp_start_steps.setdefault(int(idx), self._current_step)

        if self._ramp_start_steps:
            for idx in list(self._ramp_start_steps.keys()):
                if idx >= mask.numel() or not bool(mask[idx]):
                    del self._ramp_start_steps[idx]

        self._prev_group_mask = mask.detach().clone()

    def _compute_ramp_lambda(self, indices: List[int], device: torch.device) -> torch.Tensor:
        if not indices or self.ramp_len <= 0 or self._current_step is None:
            return torch.ones(len(indices), device=device)

        start_steps = torch.tensor(
            [self._ramp_start_steps.get(int(idx), self._current_step) for idx in indices],
            device=device,
            dtype=torch.float32,
        )
        t = (float(self._current_step) - start_steps) / float(self.ramp_len)
        if self.ramp_mode == "sigmoid":
            lam = torch.sigmoid(self.ramp_sigmoid_a * (t - 0.5))
        else:
            lam = torch.clamp(t, 0.0, 1.0)

        for idx, value in zip(indices, lam.detach().cpu().tolist()):
            if value >= 1.0 - 1e-6:
                self._ramp_start_steps.pop(int(idx), None)
        return lam

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
        weight = _to_local_if_dtensor(weight)
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

        if x.shape[-1] % self.group_size != 0:
            raise ValueError(
                f"Seq2BitQuantizer fake_quant expects last dim divisible by group_size, "
                f"got x.shape={tuple(x.shape)}, group_size={self.group_size}"
            )

        ori_shape = x.shape
        xg = x.reshape(-1, self.group_size)
        group_count = xg.shape[0]
        flat_scale = scale.reshape(-1)
        if flat_scale.numel() != group_count:
            raise ValueError(
                f"Seq2BitQuantizer scale numel mismatch: scale={flat_scale.numel()}, groups={group_count}"
            )

        ratios = self.group_ratio
        if ratios is not None:
            if ratios.numel() != group_count:
                raise ValueError(f"group_ratio length mismatch: got {ratios.numel()}, expected {group_count}")
            ratios = ratios.to(device=xg.device, dtype=xg.dtype).clamp(0.0, 1.0)
            mask = ratios > 0
            if not mask.any():
                return xg.reshape(ori_shape)

            selected_indices = torch.nonzero(mask, as_tuple=True)[0]
            x_quant = torch.index_select(xg, 0, selected_indices)
            selected_scale = torch.index_select(flat_scale, 0, selected_indices)
            x_int = self._quantize(x_quant, selected_scale, None)
            x_dequant = self._dequantize(x_int, selected_scale, None)

            selected_ratio = torch.index_select(ratios, 0, selected_indices).view(-1, 1)
            x_mix = x_quant + (x_dequant - x_quant) * selected_ratio

            out = xg.clone()
            out.index_copy_(0, selected_indices, x_mix)
            return out.reshape(ori_shape)

        mask = self.group_mask
        if mask is not None:
            if mask.numel() != group_count:
                raise ValueError(f"group_mask length mismatch: got {mask.numel()}, expected {group_count}")
            mask = mask.to(device=xg.device, dtype=torch.bool)
        else:
            quantized_groups = self._split_quant_groups(xg)
            if quantized_groups <= 0:
                return xg.reshape(ori_shape)
            if quantized_groups >= group_count:
                mask = torch.ones(group_count, device=xg.device, dtype=torch.bool)
            else:
                mask = torch.zeros(group_count, device=xg.device, dtype=torch.bool)
                mask[:quantized_groups] = True

        if mask.all() and self.interpolate_ratio <= 0.0 and self.ramp_len <= 0 and x.is_cuda:
            x_kernel = x.contiguous()
            scale_kernel = flat_scale.contiguous()
            use_kernel = (
                self.group_size in (64, 128, 256)
                and x_kernel.numel() % self.group_size == 0
                and scale_kernel.numel() == x_kernel.numel() // self.group_size
            )
            if use_kernel:
                from EfficientQAT.core.quantizer.kernel.fake_quant import fake_quant_ste_seq2bit

                return fake_quant_ste_seq2bit(x_kernel, scale_kernel, self.group_size)

        if not mask.any():
            return xg.reshape(ori_shape)

        self._update_ramp_state(mask)

        selected_indices = torch.nonzero(mask, as_tuple=True)[0]
        x_quant = torch.index_select(xg, 0, selected_indices)
        selected_scale = torch.index_select(flat_scale, 0, selected_indices)
        x_int = self._quantize(x_quant, selected_scale, None)
        x_dequant = self._dequantize(x_int, selected_scale, None)

        if self.interpolate_ratio > 0.0 and self.ramp_len <= 0:
            ratio = self.interpolate_ratio
            x_dequant = x_dequant * ratio + x_quant * (1 - ratio)

        out = xg.clone()
        out.index_copy_(0, selected_indices, x_dequant)

        if self._ramp_start_steps and self.ramp_len > 0 and self._current_step is not None:
            ramp_indices = [idx for idx in self._ramp_start_steps.keys() if idx < group_count and bool(mask[idx])]
            if ramp_indices:
                ramp_idx_tensor = torch.tensor(ramp_indices, device=xg.device, dtype=torch.long)
                x_fp = torch.index_select(xg, 0, ramp_idx_tensor)
                x_dequant_ramp = torch.index_select(out, 0, ramp_idx_tensor)
                lam = self._compute_ramp_lambda(ramp_indices, device=xg.device).view(-1, 1)
                x_act = x_fp * (1.0 - lam) + x_dequant_ramp * lam
                out.index_copy_(0, ramp_idx_tensor, x_act)
        return out.reshape(ori_shape)

    @torch.no_grad()
    def quantize_codes(self, x: torch.Tensor) -> torch.Tensor:
        scale, _ = self.cal_qparams(self.alpha, None)
        q = self._quantize(x, scale, None)
        return q.to(torch.uint8)

    @torch.no_grad()
    def update_qparams_from_weight(self, weight: torch.Tensor) -> None:
        alpha, _ = self.init_with_weight(
            weight,
            self.n_bits,
            self.group_size,
            clamp_method=self.clamp_method,
            symmetric=False,
        )
        target_alpha = _to_local_if_dtensor(self.alpha)
        alpha = alpha.to(device=target_alpha.device, dtype=target_alpha.dtype)
        aligned = _align_alpha_numel(alpha, target_alpha.numel()).view_as(target_alpha)
        target_alpha.copy_(aligned)

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
