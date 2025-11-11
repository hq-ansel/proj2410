from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import Iterable, Optional

import torch

from .block_pipeline import BlockPipeline, BlockContext, build_block_schedule
from .gptq_pipeline import gptq_pipeline
from EfficientQAT.core.pipeline import PipelineStage




class _GPTQBlockPipeline(BlockPipeline):
    """Minimal BlockPipeline wrapper for GPTQ orchestration."""

    def __init__(self, model, train_dataset, args, *, executor, logger=None) -> None:
        self._train_dataset = train_dataset
        self._gptq_args = args
        self._schedule_cache: Optional[list[PipelineStage]] = None
        super().__init__(
            model,
            SimpleNamespace(),
            trainloader=[],
            valloader=[],
            executor=executor,
            logger=logger,
            enable_loss_recorder=False,
            schedule_builder=self._schedule_builder,
        )

    def _setup(self, ctx):
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

    def _prepare_data(self, ctx):
        ctx.extras["gptq_args"] = self._gptq_args
        ctx.extras["gptq_train_dataset"] = self._train_dataset
        return ctx

    def _export(self, ctx):
        torch.cuda.empty_cache()
        gc.collect()

    def _teardown(self, ctx):
        self.model.config.use_cache = self.use_cache

    def _schedule_builder(self, ctx):
        if self._schedule_cache is None:
            num_layers = len(ctx.layers)
            self._schedule_cache = list(
                build_block_schedule(num_layers, window_size=num_layers, step=num_layers)
            )
        return self._schedule_cache


def _gptq_stage_executor(ctx: BlockContext, stage: PipelineStage) -> None:
    if ctx.extras.get("gptq_done"):
        return
    args = ctx.extras["gptq_args"]
    dataset = ctx.extras["gptq_train_dataset"]
    ctx.extras["gptq_model"] = gptq_pipeline(ctx.model, dataset, args)
    ctx.extras["gptq_done"] = True


def run_gptq_with_block_pipeline(model, train_dataset: Iterable, args, logger=None):
    """
    Draft helper that runs GPTQ quantisation inside the shared BlockPipeline so
    future integrations can reuse the same executor contract as block / greedy
    flows. Returns the quantized GPTQModel.
    """

    pipeline = _GPTQBlockPipeline(
        model=model,
        train_dataset=train_dataset,
        args=args,
        executor=_gptq_stage_executor,
        logger=logger,
    )
    ctx = pipeline.run()
    return ctx.extras.get("gptq_model")
