"""
Shared helpers for constructing quantizer instances.

Historically each quantized Linear wrapper re-implemented the logic that maps
`quantizer_version` and several feature flags (e.g., `gradual_quant`,
`iterative_freezing`, `dsq`) onto concrete classes. This module centralises the
selection so call sites can stay compact and future changes only touch one
place.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

QuantizerModule = Any
QuantizerClass = Any

_PKG_LOADERS: Dict[str, Callable[[], QuantizerModule]] = {
    "v1": lambda: importlib.import_module("EfficientQAT.quantize.quantizer"),
    "v2": lambda: importlib.import_module("EfficientQAT.quantize.quantizerv2"),
    "v3": lambda: importlib.import_module("EfficientQAT.quantize.quantizerv3"),
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
