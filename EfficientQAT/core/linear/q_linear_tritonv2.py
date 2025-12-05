from typing import Optional, Tuple

import torch

from .q_linear_autograd import QuantLinearFunction
from .q_linear_base import BACKEND, DEVICE, PLATFORM, TritonModuleMixin
from .q_linear_pack import PackableQuantLinear


class TritonV2QuantLinear(PackableQuantLinear, TritonModuleMixin):
    SUPPORTS_BITS = [2, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.XPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.int16, torch.int8]
    # for transformers/optimum tests compat
    QUANT_TYPE = "tritonv2"

    """
    Triton v2 quantized linear layer.

    Calls dequant kernel to dequantize the weights then uses torch.matmul to compute the output.
    """

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features,
        out_features,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TRITON),
            register_buffers=True,
            **kwargs,
        )

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        return cls._validate(**args)

    def post_init(self):
        super().post_init()

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        out = QuantLinearFunction.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
        ).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x.dtype)


__all__ = ["TritonV2QuantLinear"]
