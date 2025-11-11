"""
Shared helpers for constructing and exporting quantizer instances.

Historically each quantized Linear wrapper re-implemented the logic that maps
`quantizer_version` and feature flags (e.g., `gradual_quant`,
`iterative_freezing`, `dsq`) onto concrete classes and also duplicated the code
that prepares packed parameters for real-quant deployment. This module
centralises both responsibilities.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

import torch



QuantizerModule = Any
QuantizerClass = Any

_PKG_LOADERS: Dict[str, Callable[[], QuantizerModule]] = {
    "v1": lambda: importlib.import_module("EfficientQAT.quantize.quantizer"),
    "v2": lambda: importlib.import_module("EfficientQAT.quantize.quantizerv2"),
    "v3": lambda: importlib.import_module("EfficientQAT.quantize.quantizerv3"),
}

_REAL_LINEAR_CLASS_NAMES: Dict[str, str] = {
    "v1": "QuantLinear",
    "v2": "QuantLinearV2",
    "v3": "QuantLinearV3",
}

_VARIANT_RULES: Tuple[Tuple[str, str], ...] = (
    ("gradual_quant", "GradualUniformAffineQuantizer"),
    ("iterative_freezing", "GradualUniformAffineQuantizerV2"),
    ("dsq", "DSQuantizer"),
)


def _load_quantizer_module(version: str) -> QuantizerModule:
    try:
        loader = _PKG_LOADERS[version]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported quantizer version '{version}'") from exc
    return loader()


def _resolve_quantizer_class(pkg: QuantizerModule, args: Dict[str, Any]) -> QuantizerClass:
    for flag, attr in _VARIANT_RULES:
        if args.get(flag, False):
            try:
                return getattr(pkg, attr)
            except AttributeError as exc:
                raise ValueError(
                    f"Quantizer package '{pkg.__name__}' does not implement '{attr}' "
                    f"but flag '{flag}' was requested."
                ) from exc
    try:
        return pkg.UniformAffineQuantizer
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Quantizer package '{pkg.__name__}' lacks UniformAffineQuantizer") from exc


def build_weight_quantizer(
    *,
    weight,
    wbits: int,
    group_size: int,
    args: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Factory that instantiates the correct quantizer implementation according to
    the user-provided args.

    Parameters
    ----------
    weight:
        Reference weight tensor from the original Linear module.
    wbits:
        Target precision.
    group_size:
        Group size passed down to the quantizer constructor.
    args:
        Configuration dict (may include `quantizer_version`, `gradual_quant`,
        `iterative_freezing`, `dsq`, etc.).
    """

    conf: Dict[str, Any] = args.copy() if args else {}
    version = conf.get("quantizer_version", "v1")
    pkg = _load_quantizer_module(version)
    quantizer_cls = _resolve_quantizer_class(pkg, conf)
    return quantizer_cls(wbits, group_size, weight=weight, args=conf)


def _resolve_real_linear_class(version: str):
    module = importlib.import_module("EfficientQAT.quantize.int_linear_real")
    try:
        class_name = _REAL_LINEAR_CLASS_NAMES[version]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported quantizer version '{version}'") from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Real quantized linear class '{class_name}' is not available in int_linear_real.py"
        ) from exc


def build_real_quant_linear(
    *,
    version: str,
    wbits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    bias: bool,
    **kwargs,
) -> Any:
    """Create the correct packed QuantLinear implementation for deployment."""

    quant_linear_cls = _resolve_real_linear_class(version)
    return quant_linear_cls(
        wbits,
        group_size,
        in_features,
        out_features,
        bias,
        **kwargs,
    )


def export_scale_tensor(weight_quantizer) -> torch.Tensor:
    """Detach + clamp the scale tensor and return it on CPU for packing."""

    return weight_quantizer.scale.clamp(1e-4, 1e4).detach().cpu()


def export_zero_tensor(weight_quantizer, version: str) -> torch.Tensor:
    """
    Prepare the zero-point tensor for packing.

    GPTQ-style (v1) quantizers store floating zero-points that need rounding,
    whereas newer versions already maintain the correct dtype. This helper keeps
    the logic in one place.
    """

    zeros = weight_quantizer.zero_point.detach()
    if version == "v1":
        zeros = zeros.round()
    return zeros.cpu()
