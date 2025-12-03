# quantizer/greedy/pipeline.py
"""Greedy block-wise quantization pipeline that plugs into the shared runner."""

from __future__ import annotations

import copy
import gc
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers.modeling_utils import PreTrainedModel



try:
    from EfficientQAT.quantize import int_linear_fake, utils
except ImportError:  # pragma: no cover - fallback for relative import
    from .. import int_linear_fake, utils

try:
    from EfficientQAT.datautils_block import LazyLoadDatasetV2
except ImportError:  # pragma: no cover - fallback for relative import
    from ...datautils_block import LazyLoadDatasetV2

try:
    from EfficientQAT.quantize.loss_utils import get_loss_func
except ImportError:  # pragma: no cover - fallback for relative import
    from ...loss_utils import get_loss_func

try:
    from EfficientQAT.quantize.block_pipeline import (
        BlockContext,
        BlockPipeline,
        build_block_schedule,
    )
except ImportError:  # pragma: no cover - fallback for relative import
    from ..block_pipeline import BlockContext, BlockPipeline, build_block_schedule

from EfficientQAT.core.pipeline import PipelineStage
from .common import CommonInputDataset

try:
    from ..visualization import VisualizationRecorder
    VISUALIZATION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    VisualizationRecorder = None
    VISUALIZATION_AVAILABLE = False

__all__ = ["GreedyBlockPipeline", "trans_quant_block"]


def trans_quant_block(qlayer: torch.nn.Module, hyper_params: Dict) -> torch.nn.Module:
    """Replace every linear module in a layer with its quantized counterpart."""
    for name, module in qlayer.named_modules():
        if isinstance(module, torch.nn.Linear):
            quantlinear = int_linear_fake.QuantLinear(
                module,
                hyper_params["wbits"],
                hyper_params["group_size"],
                hyper_params,
            )
            quantlinear.set_quant_state(True)
            utils.set_op_by_name(qlayer, name, quantlinear)
    return qlayer


class GreedyBlockPipeline(BlockPipeline):
    """
    BlockPipeline specialisation that reuses LazyLoad datasets/sniffers used by
    the greedy trainer while delegating scheduling/execution to the shared
    runner.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: Dict[str, Any],
        trainloader,
        valloader,
        *,
        executor: Callable[[BlockContext, PipelineStage], None],
        logger: Optional[torch.jit.logging.Logger] = None,
    ) -> None:
        self.config = config
        self.train_params = config.get("train_param_settings", {})
        self.hyper_params = config.get("hyperparam_settings", {})
        self.cluster_settings = config.get("cluster_settings", {})
        self._schedule_cache: Optional[List[PipelineStage]] = None
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.use_cache: bool = model.config.use_cache
        loss_path = config.get("log_loss")
        loss_factory = (lambda: utils.BlockLossRecorder(file_path=loss_path)) if loss_path else None

        super().__init__(
            model,
            config,
            trainloader,
            valloader,
            executor=executor,
            logger=logger,
            loss_recorder_factory=loss_factory,
            enable_loss_recorder=loss_factory is not None,
            schedule_builder=self._schedule_builder,
        )

    # ------------------------------------------------------------------ Hooks
    def _setup(self, ctx):
        self.logger.info("Starting greedy pipeline ...")
        cuda_ids = self.cluster_settings.get("cuda_ids", [])
        default_cuda = cuda_ids[-1] if cuda_ids else "cuda:0"
        device = default_cuda if torch.cuda.is_available() else "cpu"
        self.train_params["dev"] = device
        self.train_params["dtype"] = torch.float16
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.model = self.model.cpu()
        ctx.extras["config"] = self.config

    def _prepare_data(self, ctx):
        train_params = self.train_params
        hyper_params = self.hyper_params
        config = self.config
        trainloader = self.trainloader
        valloader = self.valloader
        if trainloader is None or valloader is None:
            raise ValueError("Train/validation loaders must be provided before data preparation.")

        if train_params.get("quant_shedule_type") == "full":
            for idx, layer in enumerate(self.model.model.layers):
                self.model.model.layers[idx] = trans_quant_block(
                    qlayer=layer,
                    hyper_params=hyper_params,
                )
        is_quant_layer = [
            train_params.get("quant_shedule_type") == "full"
            for _ in self.model.model.layers
        ]
        ctx.extras["is_quant_layer"] = is_quant_layer

        loss_func = get_loss_func(train_params["loss_func"])
        ctx.extras["loss_func"] = loss_func

        vis_recorder = None
        if VISUALIZATION_AVAILABLE and config.get("enable_visualization", False):
            vis_type = config.get("visualization_type", "tensorboard")
            if vis_type == "tensorboard":
                log_dir = config.get("tensorboard_log_dir", "./tensorboard_logs")
                vis_recorder = VisualizationRecorder(
                    visualization_type="tensorboard",
                    log_dir=log_dir,
                    experiment_name=config.get("experiment_name", "efficientqat"),
                )
            elif vis_type == "wandb":
                vis_recorder = VisualizationRecorder(
                    visualization_type="wandb",
                    project_name=config.get("wandb_project", "efficientqat"),
                    experiment_name=config.get("experiment_name", "experiment"),
                )
        ctx.extras["vis_recorder"] = vis_recorder

        target_model = None
        if not train_params.get("with_catcher"):
            train_dataset = LazyLoadDatasetV2(
                model=self.model,
                dataloader=trainloader,
                crossblock_window_size=hyper_params["crossblock_window_size"],
                device=train_params["dev"],
            )
            val_dataset = LazyLoadDatasetV2(
                model=self.model,
                dataloader=valloader,
                crossblock_window_size=hyper_params["crossblock_window_size"],
                device=train_params["dev"],
            )
            if train_dataset.attention_mask is None:
                attention_mask = train_dataset.attention_mask
            else:
                attention_mask = train_dataset.attention_mask.to(train_params["dev"])
            position_embeddings = (
                train_dataset.position_embeddings[0].to(train_params["dev"]),
                train_dataset.position_embeddings[1].to(train_params["dev"]),
            )
            ctx.attention_mask = attention_mask
            ctx.position_embeddings = position_embeddings
        else:
            train_inputs = [batch[0] for batch in trainloader]
            val_inputs = [batch[0] for batch in valloader]
            train_dataset = CommonInputDataset(train_inputs)
            val_dataset = CommonInputDataset(val_inputs)
            ctx.attention_mask = None
            ctx.position_embeddings = None
            target_model = copy.deepcopy(self.model)
            target_model.to(train_params["dev"])
        ctx.extras["target_model"] = target_model

        ctx.extras["train_dataset"] = train_dataset
        ctx.extras["val_dataset"] = val_dataset

        self.trainloader = None
        self.valloader = None
        gc.collect()
        if self.loss_recorder_factory is not None and ctx.loss_recorder is None:
            ctx.loss_recorder = self.loss_recorder_factory()
        return ctx

    def _export(self, ctx):
        vis_recorder = ctx.extras.get("vis_recorder")
        if vis_recorder is not None:
            vis_recorder.close()
        torch.cuda.empty_cache()
        gc.collect()

    def _teardown(self, ctx):
        del ctx
        self.model.config.use_cache = self.use_cache

    # ------------------------------------------------------------------ Helpers
    def _schedule_builder(self, ctx):
        if self._schedule_cache is None:
            window = max(1, self.hyper_params.get("crossblock_window_size", 1))
            step = max(1, self.hyper_params.get("slide_step", 1))
            stages = list(build_block_schedule(len(self.layers), window_size=window, step=step))
            if self.train_params.get("train_shedule_type") == "end2start":
                stages = list(reversed(stages))
            self._schedule_cache = stages
        ctx.extras["schedule"] = self._schedule_cache
        return self._schedule_cache
