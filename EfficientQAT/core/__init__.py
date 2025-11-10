"""
Core abstractions shared across EfficientQAT components.

Currently exposes the sampling pipeline primitives so that CLI tools and
dataset builders can rely on a single implementation.
"""

from .sampling import (  # noqa: F401
    SamplerConfig,
    SamplerPipeline,
    TextSource,
    Tokenizer,
    Writer,
    FilePerLineTextSource,
    ListTextSource,
    JSONLWriter,
    Chunker,
    ReservoirSampler,
    extract_random_chunks,
)
from .quantization import (  # noqa: F401
    build_weight_quantizer,
    build_real_quant_linear,
    export_scale_tensor,
    export_zero_tensor,
)

__all__ = [
    "SamplerConfig",
    "SamplerPipeline",
    "TextSource",
    "Tokenizer",
    "Writer",
    "FilePerLineTextSource",
    "ListTextSource",
    "JSONLWriter",
    "Chunker",
    "ReservoirSampler",
    "extract_random_chunks",
    "build_weight_quantizer",
    "build_real_quant_linear",
    "export_scale_tensor",
    "export_zero_tensor",
]
