"""
Pipeline orchestration utilities shared across training/quantization scripts.

The goal is to give every entrypoint the same high-level structure:

    setup -> prepare data -> build schedule -> train/eval per stage -> export

Specific behaviours (dataset preparation, scheduling strategy, loss logic,
export format, etc.) are injected via hook callables so we can reuse the
surrounding control flow.
"""

from .base import (  # noqa: F401
    PipelineConfig,
    PipelineContext,
    PipelineHooks,
    PipelineRunner,
    PipelineStage,
)



__all__ = [
    "PipelineConfig",
    "PipelineContext",
    "PipelineHooks",
    "PipelineRunner",
    "PipelineStage",
]
