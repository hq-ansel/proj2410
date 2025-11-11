from __future__ import annotations

import gc
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from EfficientQAT.core.pipeline import (
    PipelineConfig,
    PipelineHooks,
    PipelineRunner,
    PipelineStage,
)

from .. import utils
from ..datautils_block import BlockTrainDataset

__all__ = [
    "BlockContext",
    "BlockExecutor",
    "BlockPipeline",
    "CombinedDataset",
    "update_dataset",
    "build_block_schedule",
]


class BlockExecutor(Protocol):
    def __call__(self, ctx: "BlockContext", stage: PipelineStage) -> None: ...


@dataclass
class BlockContext:
    model: nn.Module
    args: Any
    logger: logging.Logger
    device: torch.device
    dtype: torch.dtype
    layers: nn.ModuleList
    fp_train_inps: Optional[BlockTrainDataset] = None
    fp_val_inps: Optional[BlockTrainDataset] = None
    quant_train_inps: Optional[BlockTrainDataset] = None
    quant_val_inps: Optional[BlockTrainDataset] = None
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attention_mask_batch: Optional[torch.Tensor] = None
    loss_recorder: Optional[utils.BlockLossRecorder] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class CombinedDataset(Dataset):
    def __init__(self, quant_dataset: BlockTrainDataset, fp_dataset: BlockTrainDataset) -> None:
        if len(quant_dataset) != len(fp_dataset):
            raise ValueError("Datasets must have the same length")
        self.quant_dataset = quant_dataset
        self.fp_dataset = fp_dataset

    def __len__(self) -> int:
        return len(self.quant_dataset)

    def __getitem__(self, idx: int):
        return self.quant_dataset[idx], self.fp_dataset[idx]


def update_dataset(
    layer: Union[nn.Module, Sequence[nn.Module]],
    dataset: BlockTrainDataset,
    device: torch.device,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
) -> None:
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            modules = _normalize_layers(layer)
            for index, inps in enumerate(dataset):
                inps = inps.to(device)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                hidden = inps
                for module in modules:
                    hidden = module(
                        hidden,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
                dataset.update_data(index, hidden.to("cpu"))


class BlockPipeline:
    """Reusable pipeline wrapper shared by block-based quantisation scripts."""

    def __init__(
        self,
        model: nn.Module,
        args: Any,
        trainloader,
        valloader,
        *,
        executor: BlockExecutor,
        logger: Optional[logging.Logger] = None,
        loss_recorder_factory: Optional[Callable[[], utils.BlockLossRecorder]] = None,
        enable_loss_recorder: bool = True,
        schedule_builder: Optional[Callable[[BlockContext], Iterable[PipelineStage]]] = None,
    ) -> None:
        if executor is None:
            raise ValueError("BlockPipeline requires a non-null executor.")
        self.model = model
        self.args = args
        self.trainloader = trainloader
        self.valloader = valloader
        self.executor = executor
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.layers = model.model.layers
        self.use_cache = model.config.use_cache
        self.cache_paths: Dict[str, Optional[str]] = {}
        self.loss_recorder_factory = (
            (loss_recorder_factory or self._default_loss_recorder_factory) if enable_loss_recorder else None
        )
        self.schedule_builder = schedule_builder
        self.ctx = BlockContext(
            model=model,
            args=args,
            logger=self.logger,
            device=self.device,
            dtype=self.dtype,
            layers=self.layers,
        )

    def run(self) -> nn.Module:
        hooks = PipelineHooks(
            setup=self._setup,
            prepare_data=self._prepare_data,
            build_schedule=self._build_schedule,
            train_stage=self._train_stage,
            after_stage=self._after_stage,
            export=self._export,
            teardown=self._teardown,
        )
        PipelineRunner(PipelineConfig(enable_eval=False), hooks).run()
        return self.model

    # ------------------------------------------------------------------ Hooks
    def _setup(self, pipeline_ctx) -> None:
        self.logger.info("Starting ...")
        if getattr(self.args, "off_load_to_disk", False):
            self.logger.info(
                "offload the training dataset to disk, saving CPU memory, "
                "but may slowdown the training due to additional I/O..."
            )
        self.model.config.use_cache = False
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
        self.model.model.norm = self.model.model.norm.to(self.device)
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.device)
        self.layers[0] = self.layers[0].to(self.device)

    def _prepare_data(self, pipeline_ctx):
        args = self.args
        cache_root = Path(getattr(args, "cache_dir", "/tmp"))
        flag = time.time()
        if args.off_load_to_disk:
            self.cache_paths = {
                "fp_train": str(cache_root / f"{flag}" / "block_training_fp_train"),
                "fp_val": str(cache_root / f"{flag}" / "block_training_fp_val"),
                "quant_train": str(cache_root / f"{flag}" / "block_training_quant_train"),
                "quant_val": str(cache_root / f"{flag}" / "block_training_quant_val"),
            }
            for path in self.cache_paths.values():
                if path and os.path.exists(path):
                    shutil.rmtree(path)
        else:
            self.cache_paths = {key: None for key in ["fp_train", "fp_val", "quant_train", "quant_val"]}

        self.ctx.fp_train_inps = BlockTrainDataset(
            args.train_size,
            args.training_seqlen,
            self.model.config.hidden_size,
            args.batch_size,
            self.dtype,
            cache_path=self.cache_paths["fp_train"],
            off_load_to_disk=args.off_load_to_disk,
        )
        self.ctx.fp_val_inps = BlockTrainDataset(
            args.val_size,
            args.training_seqlen,
            self.model.config.hidden_size,
            args.batch_size,
            self.dtype,
            cache_path=self.cache_paths["fp_val"],
            off_load_to_disk=args.off_load_to_disk,
        )

        class Catcher(nn.Module):
            def __init__(self, module, dataset):
                super().__init__()
                self.module = module
                self.dataset = dataset
                self.index = 0
                self.attention_mask = None
                self.position_ids = None

            def forward(self, inp, **kwargs):
                self.dataset.update_data(self.index, inp.squeeze(0).to("cpu"))
                self.index += 1
                if self.attention_mask is None:
                    self.attention_mask = kwargs["attention_mask"]
                if self.position_ids is None:
                    self.position_ids = kwargs["position_ids"]
                raise ValueError

        self._catch_inputs(self.trainloader, self.ctx.fp_train_inps, Catcher)
        val_catcher = self._catch_inputs(self.valloader, self.ctx.fp_val_inps, Catcher, keep_catcher=True)

        attention_mask = val_catcher.attention_mask
        position_ids = val_catcher.position_ids
        self.layers[0] = self.layers[0].module
        if attention_mask is not None:
            self.ctx.attention_mask_batch = attention_mask.repeat(self.args.batch_size, 1, 1, 1).float()
        else:
            self.logger.info(
                "No attention mask caught from the first layer. "
                "Seems that model's attention works without a mask."
            )
            self.ctx.attention_mask_batch = None
        self.ctx.attention_mask = attention_mask
        self.ctx.position_ids = position_ids

        self.layers[0] = self.layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        self.model.model.norm = self.model.model.norm.cpu()
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        torch.cuda.empty_cache()

        if args.off_load_to_disk:
            shutil.copytree(self.cache_paths["fp_train"], self.cache_paths["quant_train"])
            shutil.copytree(self.cache_paths["fp_val"], self.cache_paths["quant_val"])

        self.ctx.quant_train_inps = BlockTrainDataset(
            args.train_size,
            args.training_seqlen,
            self.model.config.hidden_size,
            args.batch_size,
            self.dtype,
            cache_path=self.cache_paths["quant_train"],
            off_load_to_disk=args.off_load_to_disk,
        )
        self.ctx.quant_val_inps = BlockTrainDataset(
            args.val_size,
            args.training_seqlen,
            self.model.config.hidden_size,
            args.batch_size,
            self.dtype,
            cache_path=self.cache_paths["quant_val"],
            off_load_to_disk=args.off_load_to_disk,
        )

        if not args.off_load_to_disk:
            for idx, data in enumerate(self.ctx.fp_train_inps):
                self.ctx.quant_train_inps.update_data(idx, data)
            for idx, data in enumerate(self.ctx.fp_val_inps):
                self.ctx.quant_val_inps.update_data(idx, data)

        if self.loss_recorder_factory is not None and self.ctx.loss_recorder is None:
            self.ctx.loss_recorder = self.loss_recorder_factory()
        return self.ctx

    def _catch_inputs(self, dataloader, dataset, catcher_cls, keep_catcher: bool = False):
        self.layers[0] = catcher_cls(self.layers[0], dataset)
        iters = len(dataloader) // self.args.batch_size
        with torch.no_grad():
            for i in range(iters):
                data = torch.cat(
                    [dataloader[j][0] for j in range(i * self.args.batch_size, (i + 1) * self.args.batch_size)],
                    dim=0,
                )
                try:
                    self.model(data.to(self.device))
                except ValueError:
                    pass
        catcher = self.layers[0]
        if not keep_catcher:
            self.layers[0] = self.layers[0].module
        return catcher

    def _build_schedule(self, pipeline_ctx) -> Iterable[PipelineStage]:
        if self.schedule_builder is not None:
            return list(self.schedule_builder(self.ctx))
        return list(build_block_schedule(len(self.layers)))

    def _train_stage(self, pipeline_ctx, stage: PipelineStage) -> None:
        self.executor(self.ctx, stage)

    def _after_stage(self, pipeline_ctx, stage: PipelineStage) -> None:
        if self.ctx.loss_recorder is not None:
            self.ctx.loss_recorder.save_to_file()

    def _export(self, pipeline_ctx) -> None:
        if getattr(self.args, "off_load_to_disk", False):
            for path in self.cache_paths.values():
                if path and os.path.exists(path):
                    shutil.rmtree(path)
        torch.cuda.empty_cache()
        gc.collect()

    def _teardown(self, pipeline_ctx) -> None:
        self.model.config.use_cache = self.use_cache

    @staticmethod
    def _default_loss_recorder_factory() -> utils.BlockLossRecorder:
        loss_dir = Path("/home/ubuntu/data/exp/proj2410/logs")
        loss_dir.mkdir(parents=True, exist_ok=True)
        loss_path = loss_dir / "Llama2-7b-block-ap-loss.csv"
        return utils.BlockLossRecorder(file_path=str(loss_path))


def build_block_schedule(
    num_layers: int,
    *,
    window_size: int = 1,
    step: int = 1,
) -> Iterable[PipelineStage]:
    if num_layers <= 0:
        return []
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    stages: List[PipelineStage] = []
    for start in range(0, num_layers, step):
        end = min(start + window_size, num_layers)
        indices = list(range(start, end))
        name = f"block-{indices[0]}" if len(indices) == 1 else f"block-{indices[0]}-{indices[-1]}"
        stages.append(
            PipelineStage(
                name=name,
                metadata={
                    "indices": indices,
                    "start": indices[0],
                    "end": indices[-1],
                    **({"index": indices[0]} if len(indices) == 1 else {}),
                },
            )
        )
    return stages


def _normalize_layers(layer: Union[nn.Module, Sequence[nn.Module]]) -> List[nn.Module]:
    if isinstance(layer, nn.ModuleList):
        return list(layer)
    if isinstance(layer, (list, tuple)):
        return list(layer)
    return [layer]
