# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque
from contextlib import nullcontext
from typing import Callable, Deque, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from ..parallel_state import get_parallel_state
from .partition import partition_model
from .stage import PipelineStage

class PipelineRuntime(nn.Module):
    def __init__(self, model: nn.Module, input_shape: torch.Size = None):
        """
        Args:
            model: The full model (on CPU/Meta) to be partitioned.
            input_shape: Expected shape of the hidden states passed between stages. 
                         Format: (batch_size, seq_len, hidden_size).
                         Required for intermediate stages to allocate recv buffers.
        """
        super().__init__()
        self.ps = get_parallel_state()
        
        # 1. Partition the model into a local stage
        # This returns a LlamaPipelineStage which contains only local layers
        self.local_partition = partition_model(model)
        
        # 2. Wrap in PipelineStage to handle P2P comms
        # PipelineStage adds recv_forward() at start and send_forward() at end
        self.pipeline_stage = PipelineStage(
            self.local_partition, 
            input_shape=input_shape
        )
        
    def forward(self, *args, **kwargs):
        # The external training loop calls model(input_ids, ...)
        # Rank 0 uses args/kwargs (inputs).
        # Other ranks ignore args/kwargs (usually) and wait for P2P recv.
        
        if self.ps.is_first_pp_stage:
            # Rank 0: Use actual inputs
            return self.pipeline_stage(*args, **kwargs)
        else:
            # Middle/Last Ranks: Inputs from P2P
            return self.pipeline_stage(*args, **kwargs)

    def backward_step(self, loss=None):
        """
        Execute the backward pass logic for the pipeline.
        Must be called after forward().
        Args:
            loss: The loss value (scalar). Only required on the last pipeline stage.
        """
        self.pipeline_stage.backward_step(loss)

    def forward_backward_1f1b(
        self,
        micro_batches: List[Dict[str, torch.Tensor]],
        model_fwd_context=None,
        model_bwd_context=None,
        use_cache: bool = False,
        loss_fn: Optional[Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    ) -> float:
        """
        Simple 1F1B schedule for pipeline parallelism.
        Args:
            micro_batches: list of micro-batch dicts.
            model_fwd_context: forward context manager (e.g., autocast).
            model_bwd_context: backward context manager.
            use_cache: forwarded to model forward for compatibility.
            loss_fn: optional callable to compute loss from output and micro_batch (last stage only).
        Returns:
            total_loss (scaled by micro-batch count) on last stage, 0 otherwise.
        """
        if model_fwd_context is None:
            model_fwd_context = nullcontext()
        if model_bwd_context is None:
            model_bwd_context = nullcontext()

        if not micro_batches:
            return 0.0

        micro_batches = self._prepare_micro_batches(micro_batches)
        num_micro_batches = len(micro_batches)

        warmup = min(self.ps.pp_size - self.ps.pp_rank - 1, num_micro_batches)
        losses: Deque[torch.Tensor] = deque()
        total_loss = 0.0

        # Warmup phase (forward only)
        for idx in range(warmup):
            with model_fwd_context:
                output = self.pipeline_stage(**micro_batches[idx], use_cache=use_cache)
            if self.ps.is_last_pp_stage:
                loss = self._compute_loss(output, micro_batches[idx], loss_fn) / num_micro_batches
                losses.append(loss)

        # Steady state (1F1B)
        for idx in range(warmup, num_micro_batches):
            with model_fwd_context:
                output = self.pipeline_stage(**micro_batches[idx], use_cache=use_cache)
            if self.ps.is_last_pp_stage:
                loss = self._compute_loss(output, micro_batches[idx], loss_fn) / num_micro_batches
                losses.append(loss)

            with model_bwd_context:
                if self.ps.is_last_pp_stage:
                    loss_to_bwd = losses.popleft()
                    total_loss += loss_to_bwd.item()
                    self.pipeline_stage.backward_step(loss_to_bwd)
                else:
                    self.pipeline_stage.backward_step(None)

        # Cooldown phase (backward only)
        for _ in range(warmup):
            with model_bwd_context:
                if self.ps.is_last_pp_stage:
                    loss_to_bwd = losses.popleft()
                    total_loss += loss_to_bwd.item()
                    self.pipeline_stage.backward_step(loss_to_bwd)
                else:
                    self.pipeline_stage.backward_step(None)

        return total_loss

    def _prepare_micro_batches(self, micro_batches: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        device = torch.device(f"cuda:{self.ps.local_rank}")
        pp_group = self.ps.pp_mesh.get_group() if self.ps.pp_enabled else None
        tp_group = self.ps.tp_mesh.get_group() if self.ps.tp_enabled else None
        prepared: List[Dict[str, torch.Tensor]] = []

        for micro_batch in micro_batches:
            local = {}
            for k, v in micro_batch.items():
                if isinstance(v, torch.Tensor):
                    local[k] = v.to(device, non_blocking=True)
                else:
                    local[k] = v

            # Broadcast each tensor from pp_rank 0 to other pipeline stages (per-TP group)
            if self.ps.pp_enabled:
                for v in local.values():
                    if isinstance(v, torch.Tensor):
                        dist.broadcast(v, src=0, group=pp_group)

            # Ensure TP ranks receive identical inputs
            if self.ps.tp_enabled:
                for v in local.values():
                    if isinstance(v, torch.Tensor):
                        dist.broadcast(v, src=0, group=tp_group)

            prepared.append(local)

        return prepared

    def _compute_loss(
        self,
        output,
        micro_batch: Dict[str, torch.Tensor],
        loss_fn: Optional[Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    ) -> torch.Tensor:
        if loss_fn is not None:
            return loss_fn(output, micro_batch)
        return self._compute_lm_loss(output, micro_batch)

    def _compute_lm_loss(self, output, micro_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = micro_batch.get("labels", None)
        if labels is None:
            raise ValueError("labels must be provided for pipeline loss computation.")

        logits = output[0] if isinstance(output, tuple) else output
        if logits is None:
            raise ValueError("Pipeline output is None; cannot compute loss.")

        logits = self._gather_logits_if_tp(logits)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def _gather_logits_if_tp(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.ps.tp_enabled:
            return logits
        group = self.ps.tp_mesh.get_group()
        world_size = self.ps.tp_size
        gather_list = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(gather_list, logits, group=group)
        return torch.cat(gather_list, dim=-1)


def infer_pp_input_shape(
    model: nn.Module,
    micro_batch_size: int,
    max_seq_len: int,
    tp_size: int = 1,
    tp_shard_dim: int = -1,
) -> torch.Size:
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        hidden_sizes = getattr(getattr(model, "config", None), "hidden_sizes", None)
        if hidden_sizes:
            hidden_size = hidden_sizes[-1]
    if hidden_size is None:
        raise ValueError("Unable to infer hidden_size from model config for pipeline input shape.")

    if tp_size > 1:
        if tp_shard_dim not in (-1, 2):
            raise ValueError("Only last-dim sharding is supported for PP input shape inference.")
        if hidden_size % tp_size != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by tp_size {tp_size}.")
        hidden_size = hidden_size // tp_size

    return torch.Size([micro_batch_size, max_seq_len, hidden_size])
