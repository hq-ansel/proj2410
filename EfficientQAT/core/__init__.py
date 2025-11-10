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
]
