# core/pipeline/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Protocol




class StageHook(Protocol):
    def __call__(self, ctx: "PipelineContext", stage: "PipelineStage") -> Any: ...


class SimpleHook(Protocol):
    def __call__(self, ctx: "PipelineContext") -> Any: ...


@dataclass
class PipelineConfig:
    """Generic configuration shared by pipeline runners."""

    enable_eval: bool = True
    fail_fast: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStage:
    """Represents one unit of work in the pipeline schedule."""

    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable context passed to every hook."""

    config: PipelineConfig
    state: Dict[str, Any] = field(default_factory=dict)
    data: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)  # ✅ 改成 dict 而不是 None

@dataclass
class PipelineHooks:
    """Collection of hook callables that customise each stage."""

    setup: Optional[SimpleHook] = None
    prepare_data: Optional[SimpleHook] = None
    build_schedule: Optional[Callable[[PipelineContext], Iterable[PipelineStage]]] = None
    before_stage: Optional[StageHook] = None
    train_stage: Optional[StageHook] = None
    after_stage: Optional[StageHook] = None
    evaluate_stage: Optional[StageHook] = None
    export: Optional[SimpleHook] = None
    teardown: Optional[SimpleHook] = None


class PipelineRunner:
    """
    Template-style orchestrator that wires hook implementations into a reusable
    control flow. Scripts can provide only the hooks they need.
    """

    def __init__(self, config: PipelineConfig, hooks: PipelineHooks) -> None:
        self.config = config
        self.hooks = hooks

    def run(self, ctx: Optional[PipelineContext] = None) -> PipelineContext:
        # 如果调用方没有提供，就用默认的 PipelineContext
        if ctx is None:
            ctx = PipelineContext(config=self.config, extras=self.config.extra)

        self._call(self.hooks.setup, ctx)
        ctx = self._call(self.hooks.prepare_data, ctx)
        assert ctx.extras["train_dataset"] is not None ,f"ctx:{ctx}"
        schedule = list(self._iter_schedule(ctx))
        for stage in schedule:
            ctx.state["current_stage"] = stage
            self._call(self.hooks.before_stage, ctx, stage)
            try:
                self._call(self.hooks.train_stage, ctx, stage)
            except Exception:  # pragma: no cover - surfaces fail_fast semantic
                if self.config.fail_fast:
                    self._call(self.hooks.teardown, ctx)
                    raise
                raise
            self._call(self.hooks.after_stage, ctx, stage)
            if self.config.enable_eval:
                self._call(self.hooks.evaluate_stage, ctx, stage)

        self._call(self.hooks.export, ctx)
        self._call(self.hooks.teardown, ctx)
        return ctx

    def _iter_schedule(self, ctx: PipelineContext) -> Iterable[PipelineStage]:
        if self.hooks.build_schedule is None:
            raise ValueError("PipelineRunner requires a build_schedule hook.")
        return self.hooks.build_schedule(ctx)

    @staticmethod
    def _call(hook: Optional[Callable], *args: Any) -> Any:
        if hook is None:
            return None
        return hook(*args)
