import math
import sys
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class DEVICE(str, Enum):
    ALL = "all"  # All device
    CPU = "cpu"  # All CPU: Optimized for IPEX is CPU has AVX, AVX512, AMX, or XMX instructions
    CUDA = "cuda"  # Nvidia GPU: Optimized for Ampere+
    XPU = "xpu"  # Intel GPU: Datacenter Max + Arc
    MPS = "mps"  # MacOS GPU: Apple Silion/Metal)
    ROCM = "rocm"  # AMD GPU: ROCm maps to fake cuda

    def to_device_map(self):
        return {"": DEVICE.CUDA if self == DEVICE.ROCM else self}


class PLATFORM(str, Enum):
    ALL = "all"  # All platform
    LINUX = "linux"  # linux
    WIN32 = "win32"  # windows
    DARWIN = "darwin"  # macos


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable"  # choose the optimal trainable local kernel for post-quant training
    CUDA = "cuda"  # OK: Performance same as Torch for most cases
    TORCH = "torch"  # GOOD: about 80% of triton
    TRITON = "triton"  # VERY GOOD: all-around kernel
    EXLLAMA_V1 = "exllama_v1"  # FAST: optimized for batching == 1
    EXLLAMA_V2 = "exllama_v2"  # FASTER: optimized for batching > 1
    EXLLAMA_EORA = "exllama_eora"
    MARLIN = "marlin"  # FASTEST: marlin reduce ops in fp32 (higher precision -> more accurate, slightly slower)
    MARLIN_FP16 = "marlin_fp16"  # FASTEST and then some: marlin reduce ops in fp16 (lower precision -> less accurate, slightly faster)
    BITBLAS = "bitblas"  # EXTREMELY FAST: speed at the cost of 10+ minutes of AOT (ahead of time compilation with disk cache)
    IPEX = "ipex"  # Best kernel for Intel XPU and Intel/AMD CPU with AVX512, AMX, XMX
    VLLM = "vllm"  # External inference engine: CUDA + ROCm + IPEX
    SGLANG = "sglang"  # External inference engine: CUDA + ROCm
    MLX = "mlx"  # External inference engine: Apple MLX on M1+ (Apple Silicon)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose: bool = False, seqlen: int = 2048):
        pass


class BaseQuantLinear(nn.Module):
    SUPPORTS_BITS: List[int] = None
    SUPPORTS_GROUP_SIZE: List[int] = None
    SUPPORTS_DESC_ACT: List[bool] = None
    SUPPORTS_SYM: List[bool] = None
    SUPPORTS_SHARDS: bool = None
    SUPPORTS_TRAINING: bool = None
    SUPPORTS_AUTO_PADDING: bool = None
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: List[int] = None
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: List[int] = None

    SUPPORTS_PACK_DTYPES: List[torch.dtype] = None
    SUPPORTS_DEVICES: List[DEVICE] = None
    SUPPORTS_PLATFORM: List[PLATFORM] = None
    SUPPORTS_ADAPTERS: List[type] = []

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool,
        pack_dtype: torch.dtype,
        backend: BACKEND,
        name: str = None,
        register_buffers: bool = False,
        register_buffers_in_features: int = None,
        register_buffers_out_features: int = None,
        **kwargs,
    ):
        super().__init__()
        if name is None:
            name = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self.name = name  # full path module name in model weights
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features
        self.bits = bits
        self.desc_act = desc_act
        self.pack_dtype = pack_dtype
        self.backend = backend
        self.maxq = 2**self.bits - 1
        self.pack_dtype = pack_dtype
        self.adapter = None
        self.optimized = False

        if self.pack_dtype == torch.int8:
            self.pack_dtype_bits = 8
            self.pack_np_dtype = np.int8  # qweight saved dtype
            self.pack_np_math_dtype = np.uint8  # pre-save math dtype
        elif self.pack_dtype == torch.int16:
            self.pack_dtype_bits = 16
            self.pack_np_dtype = np.int16
            self.pack_np_math_dtype = np.uint16
        elif self.pack_dtype == torch.int32:
            self.pack_dtype_bits = 32
            self.pack_np_dtype = np.int32
            self.pack_np_math_dtype = np.uint32
        elif self.pack_dtype == torch.int64:
            self.pack_dtype_bits = 64
            self.pack_np_dtype = np.int64
            self.pack_np_math_dtype = np.uint64
        else:
            raise ValueError("Unsupported weight_dtype. Only int16 and int32 are supported.")

        # pack_factor is only used for bits 2, 4, and 8. bit3 3 does not use this variable.
        self.pack_factor = self.pack_dtype_bits // self.bits
        _, err = self._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=pack_dtype,
        )
        if err:
            raise err

        # most kernels share same buffers so they can share same register buffer code
        if register_buffers:
            # some kernels auto-pads in/out features
            in_features = self.in_features if not register_buffers_in_features else register_buffers_in_features
            out_features = self.out_features if not register_buffers_out_features else register_buffers_out_features

            self.register_buffer(
                "qweight",
                torch.zeros((in_features // self.pack_dtype_bits * self.bits, out_features), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (
                        math.ceil(in_features / self.group_size),
                        out_features // self.pack_dtype_bits * self.bits,
                    ),
                    dtype=self.pack_dtype,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    (math.ceil(in_features / self.group_size), out_features),
                    dtype=torch.float16,  # Scales are always float16
                ),
            )
            self.register_buffer(
                "g_idx",
                torch.tensor([i // self.group_size for i in range(in_features)], dtype=torch.int32),
            )
            if bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
            else:
                self.bias = None

    def post_init(self, **kwargs):
        # placeholder for subclasses that need extra initialization
        pass

    @classmethod
    def validate(
        cls,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int = None,
        out_features: int = None,
        pack_dtype: torch.dtype = None,
        dynamic: Optional[dict] = None,
        device: Optional["DEVICE"] = None,
        trainable: Optional[bool] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        return cls._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=pack_dtype,
            dynamic=dynamic,
            device=device,
            trainable=trainable,
        )

    @classmethod
    def verify_supports_params(cls):
        """
        Validate that SUPPORTS parameters are not None or empty lists, raising an exception if the validation fails.
        """
        base_supports_variables = [
            (name, value) for name, value in BaseQuantLinear.__dict__.items() if name.startswith("SUPPORTS") and not callable(value)
        ]
        child_supports_variables = [
            (name, value) for name, value in cls.__dict__.items() if name.startswith("SUPPORTS") and not callable(value)
        ]

        base_supports_variables.sort(key=lambda x: x[0])
        child_supports_variables.sort(key=lambda x: x[0])

        base_variable_names = {name for name, value in base_supports_variables}
        child_variable_names = {name for name, value in child_supports_variables}

        missing_variables = base_variable_names - child_variable_names

        if missing_variables:
            raise ValueError(f"{cls.__name__} these SUPPORTS variables are not overridden: {', '.join(sorted(missing_variables))}")

        for name, value in child_supports_variables:
            if not name.startswith("SUPPORTS") or callable(value):
                continue
            if value is None:
                raise ValueError(f"{cls.__name__}.{name} cannot be None.")

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = False,
        pack_dtype: torch.dtype = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional["DEVICE"] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[object] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        cls.verify_supports_params()

        if adapter is not None and adapter.__class__ not in cls.SUPPORTS_ADAPTERS:
            err = f"{cls} does not support adapter: {adapter}"
            return False, NotImplementedError(err)

        if pack_dtype not in cls.SUPPORTS_PACK_DTYPES:
            err = f"{cls} does not support `pack_dtype`: {pack_dtype}"
            return False, NotImplementedError(err)

        if PLATFORM.ALL not in cls.SUPPORTS_PLATFORM and sys.platform not in cls.SUPPORTS_PLATFORM:
            err = f"{cls} does not support platform: {sys.platform}"
            return False, NotImplementedError(err)

        if DEVICE.ALL not in cls.SUPPORTS_DEVICES and device is not None:
            try:
                cls.validate_device(device)
            except NotImplementedError:
                e = f"{cls} does not support device: {device}"
                return False, NotImplementedError(e)

        if trainable and not cls.SUPPORTS_TRAINING:
            err = f"{cls} does not support training."
            return False, NotImplementedError(err)

        if bits not in cls.SUPPORTS_BITS:
            err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual bits = `{bits}`"
            return False, NotImplementedError(err)
        # valid group size is set of cls.SUPPORTS_GROUP_SIZE + in_features; group_size = -1 is alias for group_size == in_features
        if group_size not in cls.SUPPORTS_GROUP_SIZE and group_size != in_features:
            err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}`"
            return False, NotImplementedError(err)
        if sym not in cls.SUPPORTS_SYM:
            err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}`"
            return False, NotImplementedError(err)
        if desc_act not in cls.SUPPORTS_DESC_ACT:
            err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}`"
            return False, NotImplementedError(err)
        if dynamic is not None:
            dynamic_bits = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_bits[pattern] = pattern_dict.get("bits", bits)
            if len(cls.SUPPORTS_BITS) == 1:
                err = f"{cls} not supported dynamic_bits, only support `{cls.SUPPORTS_BITS}` bits"
                return False, NotImplementedError(err)
            else:
                for layer, bits in dynamic_bits.items():
                    if bits not in cls.SUPPORTS_BITS:
                        err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual dynamic_bits = `{bits}` for layer `{layer}`"
                        return False, NotImplementedError(err)

            dynamic_group_size = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_group_size[pattern] = pattern_dict.get("group_size", group_size)
            for layer, group_size in dynamic_group_size.items():
                if group_size not in cls.SUPPORTS_GROUP_SIZE:
                    err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}` for layer `{layer}`"
                    return False, NotImplementedError(err)

            dynamic_sym = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_sym[pattern] = pattern_dict.get("sym", sym)
            for layer, sym in dynamic_sym.items():
                if sym not in cls.SUPPORTS_SYM:
                    err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}` for layer `{layer}`"
                    return False, NotImplementedError(err)

            dynamic_desc_act = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_desc_act[pattern] = pattern_dict.get("desc_act", desc_act)
            for layer, desc_act in dynamic_desc_act.items():
                if desc_act not in cls.SUPPORTS_DESC_ACT:
                    err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}` for layer `{layer}`"
                    return False, NotImplementedError(err)

        if in_features is not None:
            validate = all(in_features % in_fea == 0 for in_fea in cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `in_features` must be divisible by {cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)

            validate = in_features % group_size == 0 or cls.SUPPORTS_AUTO_PADDING
            if not validate:
                err = f"{cls}: `in_features` must be divisible by `group_size: {group_size}`."
                return False, NotImplementedError(err)
        if out_features is not None:
            validate = all(out_features % out_fea == 0 for out_fea in cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `out_features` must be divisible by {cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)
        return True, None

    @classmethod
    def validate_device(cls, device: "DEVICE"):
        assert isinstance(device, DEVICE)

        if device not in cls.SUPPORTS_DEVICES:
            raise NotImplementedError(f"{cls} only supports `{cls.SUPPORTS_DEVICES}`: actual device = `{device}`")

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        # override me, to perform any torch.compile logic on the kernel pre forward
        self.optimized = True
        pass


__all__ = [
    "DEVICE",
    "PLATFORM",
    "BACKEND",
    "TritonModuleMixin",
    "BaseQuantLinear",
]
