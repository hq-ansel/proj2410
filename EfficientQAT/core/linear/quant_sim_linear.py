from typing import Optional

import torch
import torch.nn as nn


class QuantSimLinear(nn.Module):
    """
    Lightweight packed-quant simulation linear layer.

    This module is intended for simulation/dequant inference paths and is kept
    separate from TritonV2QuantLinear so mixed checkpoints can explicitly route
    seq2bit modules without entering Triton packing kernel logic.
    """

    QUANT_TYPE = "quant_sim_linear"

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bits: int = 2,
        group_size: int = 64,
        impl: str = "seq2bit",
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.impl = str(impl)

        self.register_buffer("qweight", torch.empty((0, 0), dtype=torch.int32))
        self.register_buffer("scales", torch.empty((0, 0), dtype=torch.float16))
        self.register_buffer("g_idx", torch.empty((0,), dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.float16))
        else:
            self.bias = None

    @staticmethod
    def _unpack_qweight_rowwise(qweight: torch.Tensor, in_features: int, bits: int) -> torch.Tensor:
        if bits <= 0 or bits > 8:
            raise ValueError(f"Unsupported bits for unpacking: {bits}")
        pack_factor = 32 // bits
        qweight_i32 = qweight.to(dtype=torch.int32)
        unpacked = [((qweight_i32 >> (bits * i)) & ((1 << bits) - 1)) for i in range(pack_factor)]
        codes = torch.stack(unpacked, dim=-1).reshape(qweight_i32.shape[0], -1)
        if codes.shape[1] > in_features:
            codes = codes[:, :in_features]
        return codes.contiguous()

    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if self.impl != "seq2bit":
            raise NotImplementedError(f"Unsupported quant-sim impl: {self.impl}")
        if self.bits != 2:
            raise ValueError(f"seq2bit quant-sim expects bits=2, got {self.bits}")
        # Support both storage forms:
        # 1) raw codes: [out_features, in_features] (no packing, preferred for debugging)
        # 2) packed codes: [out_features, in_features / (32/bits)] (legacy packed export)
        if self.qweight.ndim == 2 and self.qweight.shape[1] == self.in_features:
            codes = self.qweight.to(dtype=torch.float32)
        else:
            codes = self._unpack_qweight_rowwise(
                self.qweight, in_features=self.in_features, bits=self.bits
            ).to(dtype=torch.float32)
        levels = codes * 0.5 - 0.75

        g_idx = self.g_idx.to(dtype=torch.long).view(1, -1)
        scales = self.scales.to(dtype=torch.float32)
        alpha = torch.gather(scales, 1, g_idx.expand(scales.shape[0], -1))
        weight = levels * alpha
        if dtype is not None:
            weight = weight.to(dtype=dtype)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x_2d = x.reshape(-1, x.shape[-1])

        weight = self.dequantize_weight(dtype=x.dtype)
        out = x_2d.matmul(weight.t())
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out.reshape(out_shape)


__all__ = ["QuantSimLinear"]
