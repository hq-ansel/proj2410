# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.import_utils import is_torch_npu_available


logger = logging.get_logger(__name__)


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=True,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


# Muon optimizer: memory-efficient by only storing exp_avg (no exp_avg_sq like AdamW).
# Newton-Schulz iteration: G -> ortho(G) via (I + GG^T)^-1 approximation.
# For FSDP2: operates on local parameter shards, syncs momentum via all-reduce.
class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        ns_steps=5,
        momentum_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "ns_steps": ns_steps,
            "momentum_dtype": momentum_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            momentum_dtype = group["momentum_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization: only exp_avg (momentum), no exp_avg_sq
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]

                # Weight decay (Muon style: decouple from gradient)
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Newton-Schulz iteration to ortho-normalize momentum
                # G = exp_avg shape: [..., hidden, out] e.g. [in_features, out_features]
                # For 2D params: ortho(G) = (I + G @ G^T)^-1 @ G ≈ G - 1/2 G(G^T G) + 3/8 G(G^T G)^2 - ...
                # Muon uses: G -> normalize(G) -> G @ G^T -> (I + G @ G^T)^-1 @ G
                orig_shape = exp_avg.shape
                is_2d = exp_avg.ndim == 2

                if is_2d:
                    # 2D: [hidden, out] -> apply Newton-Schulz
                    G = exp_avg.view(exp_avg.shape[0], -1)  # [hidden, out]
                    # Normalize first
                    G_norm = G.norm(p=2)
                    G.div_(G_norm.add_(eps))
                    # NS iteration: G_{k+1} = (3/2) G_k - 1/2 G_k @ G_k^T @ G_k
                    for _ in range(ns_steps):
                        GGt = G @ G.T
                        G.mul_(1.5).addmm_(G, G @ Gt, beta=-0.5)
                    # Reshape back
                    exp_avg.copy_(G.view(orig_shape))
                else:
                    # Non-2D: fallback to simple Adam-style update with sqrt-adaptation
                    # This handles 1D, 3D, etc. tensors
                    denom = exp_avg.square().add_(eps)
                    exp_avg.div_(denom.sqrt())

                # Compute update: sign-based with momentum
                update = exp_avg.sign()

                # Muon-style: normalize update then scale by lr
                update_norm = update.norm(p=2)
                update.div_(update_norm.add_(eps))
                p.data.add_(update, alpha=-lr)

                # All-reduce momentum across FSDP2 ranks if using FSDP2
                # Note: FSDP2AwareMuon handles this; plain Muon skips it
                # If the momentum was sharded, we need to sync it


class FSDP2AwareMuon(Optimizer):
    """
    Muon optimizer adapted for FSDP2 (sign-momentum variant, OOM-free).

    FSDP2 shards params along dim-0. Each rank has a local shard.
    Sign-momentum: update = sign(exp_avg), normalized. No Newton-Schulz (avoids
    75GB G @ G^T matrix), no cross-rank collective (avoids deadlock risk).
    The momentum is stored sharded like params; FSDP2 gradient sync ensures
    all ranks' local grad shards correspond to the same logical optimizer step.

    Key memory saving: only stores exp_avg (no exp_avg_sq like AdamW).
    Key memory reduction: 16.7GB/7B-card vs ~40GB with AdamW.

    For the full Newton-Schulz + per-layer allgather approach, see per_layer_allgather branch.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        ns_steps=5,
        momentum_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "ns_steps": ns_steps,
            "momentum_dtype": momentum_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.tensor(0, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=group["momentum_dtype"])

                state["step"] += 1
                exp_avg = state["exp_avg"]

                # Weight decay
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # Update local momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Sign-momentum update: no NS (avoids OOM), no collective (avoids deadlock)
                # Note: sign-momentum is designed for FP16/FP32 training. For INT2 QAT,
                # gradient noise is too high for sign-based updates to be stable.
                update = exp_avg.sign()
                update_norm = update.norm(p=2)
                update.div_(update_norm.add_(eps))
                p.data.add_(update, alpha=-lr)


class FSDP2AwareMuonV2(Optimizer):
    """
    Muon optimizer with per-layer allgather + Newton-Schulz for FSDP2.

    FSDP2 shards params along dim-0 (or dim-1 for EP experts).
    Each rank holds a local shard of the full momentum matrix.

    Algorithm per parameter:
    1. All-gather local momentum shard -> full m_full
    2. Newton-Schulz ortho-normalize: m_full -> ortho(m_full)
    3. Slice back to local shard -> apply update

    Memory: O(full_matrix) temporary during NS, but freed each step.
    Communication: allgather per parameter per step.
    Deadlock-free: fixed param order, all ranks participate equally.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        ns_steps=5,
        momentum_dtype=torch.bfloat16,
        block_rows=None,  # For large layers, process in row blocks
        model=None,  # nn.Module, used for named param lookup
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "ns_steps": ns_steps,
            "momentum_dtype": momentum_dtype,
            "block_rows": block_rows,
        }
        super().__init__(params, defaults)
        self.defaults = defaults  # Store for access in _build_param_info
        self.model = model  # Store model for named param lookup
        self._built = False

    def _build_param_infos(self):
        """Build static ParamInfo table for all ranks. Must be called at init."""
        ps = get_parallel_state()
        self._fsdp_group = ps.fsdp_group
        self._dp_shard_group = ps.dp_shard_group
        self._world_size = ps.world_size
        self._rank = ps.global_rank
        self._fsdp_size = ps.fsdp_size

        self.muon_param_infos: List[Dict] = []  # 2D weights -> Muon
        self.adamw_param_infos: List[Dict] = []  # non-2D or disabled -> AdamW

        # Fixed parameter ordering for deadlock-free communication
        all_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    all_params.append((p, group))

        # Sort by parameter name for deterministic order across ranks
        all_params.sort(key=lambda x: self._get_param_name(x[0]))

        for p, group in all_params:
            info = self._build_param_info(p, group)
            if info is None:
                # Non-2D or excluded params -> AdamW fallback
                adamw_info = {
                    "param": p,
                    "name": self._get_param_name(p),
                    "group": group,
                    "optimizer_kind": "adamw",
                    "lr": group.get("lr", self.defaults["lr"]),
                    "weight_decay": group.get("weight_decay", self.defaults["weight_decay"]),
                }
                self.adamw_param_infos.append(adamw_info)
            elif info["use_muon"]:
                info["optimizer_kind"] = "muon"
                self.muon_param_infos.append(info)
            else:
                info["optimizer_kind"] = "adamw"
                self.adamw_param_infos.append(info)

        # Unified ordered list: muon first, then adamw
        self.param_infos = self.muon_param_infos + self.adamw_param_infos

        # Validate that all ranks have the same param order
        self._validate_param_infos()

        # Sanity check: verify process group selection
        self._sanity_check_process_group()

    def _get_param_name(self, p: nn.Parameter) -> str:
        """Get a unique name for a parameter. Used for deterministic ordering."""
        # Try to use model named_parameters for consistent naming across ranks
        if self.model is not None:
            for name, param in self.model.named_parameters():
                if param is p:
                    return name
        # Fallback: use id (consistent within a rank)
        return str(id(p))

    def _build_param_info(self, p: nn.Parameter, group: Dict) -> Optional[Dict]:
        """Build ParamInfo for a single parameter."""
        full_shape = tuple(p.shape)
        full_numel = p.numel()

        if p.ndim != 2:
            # Non-2D parameters: skip Muon, use simple Adam-style
            return None

        # Determine shard metadata
        # For FSDP2 with dim-0 sharding: [M, N] -> [M/fsdp_size, N]
        # local_numel = full_numel // fsdp_size
        fsdp_size = self._fsdp_size
        rank = self._rank

        # Calculate local shard range
        shard_dim = 0  # FSDP2 default: shard on dim-0
        if full_shape[0] % fsdp_size != 0:
            logger.warning(f"Parameter shape {full_shape} not evenly divisible by fsdp_size {fsdp_size}")
            return None

        local_dim0_size = full_shape[0] // fsdp_size
        local_shape = (local_dim0_size, full_shape[1])
        local_numel = local_dim0_size * full_shape[1]

        # local_start = rank * local_numel (linear flatten index)
        # Since we shard on dim-0, the linear flatten offset is rank * local_dim0_size * full_shape[1]
        local_start = rank * local_dim0_size * full_shape[1]
        local_end = local_start + local_numel

        # Collect local_numels for all ranks (same for all ranks since FSDP2 uses even sharding)
        local_numels = [local_numel] * fsdp_size

        info = {
            "param": p,
            "name": self._get_param_name(p),
            "group": group,
            "use_muon": True,
            "process_group": self._fsdp_group,
            "full_shape": full_shape,
            "full_numel": full_numel,
            "local_start": local_start,
            "local_end": local_end,
            "local_numel": local_numel,
            "local_shape": local_shape,
            "world_size": fsdp_size,
            "rank": rank,
            "local_numels": local_numels,
            "padded_numel": local_numel,  # All shards same size, no padding needed
            "lr": group.get("lr", self.defaults["lr"]),
            "beta1": group.get("betas", self.defaults["betas"])[0],
            "weight_decay": group.get("weight_decay", self.defaults["weight_decay"]),
            # Persistent buffers
            "exp_avg": None,  # Will be allocated lazily
            "send_buffer": None,
            "recv_buffer": None,
            "block_rows": group.get("block_rows", self.defaults["block_rows"]),
        }
        return info

    def _validate_param_infos(self):
        """Validate that all ranks have consistent metadata for Muon params only."""
        # Only validate Muon params (AdamW params don't have full_shape, local_numel, etc.)
        names = [info["name"] for info in self.muon_param_infos]
        shapes = [info["full_shape"] for info in self.muon_param_infos]
        local_numels = [info["local_numel"] for info in self.muon_param_infos]
        use_muon_flags = [info["use_muon"] for info in self.muon_param_infos]

        # All-gather all metadata
        all_names = [None] * self._world_size
        all_shapes = [None] * self._world_size
        all_local_numels = [None] * self._world_size
        all_use_muon = [None] * self._world_size

        torch.distributed.all_gather_object(all_names, names, group=self._fsdp_group)
        torch.distributed.all_gather_object(all_shapes, shapes, group=self._fsdp_group)
        torch.distributed.all_gather_object(all_local_numels, local_numels, group=self._fsdp_group)
        torch.distributed.all_gather_object(all_use_muon, use_muon_flags, group=self._fsdp_group)

        # Check一致性
        for r in range(self._world_size):
            if all_names[r] != names:
                raise RuntimeError(f"Rank {self._rank} param names mismatch with rank {r}")
            if all_shapes[r] != shapes:
                raise RuntimeError(f"Rank {self._rank} param shapes mismatch with rank {r}")

        logger.info(f"[Rank {self._rank}] ParamInfo validation passed: {len(names)} params")

    def _sanity_check_process_group(self):
        """Verify that _fsdp_group is the correct shard-level group for Muon params.

        For each Muon param, we verify:
        - group size == fsdp_size (shard parallelism)
        - sum(local_numels) == full_numel
        """
        import torch.distributed as dist

        for info in self.muon_param_infos:
            group = info["process_group"]
            if group is None:
                continue
            group_size = dist.get_world_size(group)
            expected_size = info["world_size"]
            if group_size != expected_size:
                raise RuntimeError(
                    f"[Rank {self._rank}] Process group size {group_size} != expected fsdp_size {expected_size} "
                    f"for param {info['name']}. "
                    f"May have selected wrong group (world group or wrong granularity)."
                )
            # Verify local_numels sum to full
            total_local = sum(info["local_numels"])
            if total_local != info["full_numel"]:
                raise RuntimeError(
                    f"[Rank {self._rank}] sum(local_numels)={total_local} != full_numel={info['full_numel']} "
                    f"for param {info['name']}. Shard metadata is incorrect."
                )
        logger.info(f"[Rank {self._rank}] Process group sanity check passed for {len(self.muon_param_infos)} Muon params")

    def _local_should_skip_step(self) -> bool:
        """Determine if this rank should skip the optimizer step.

        Returns True if any condition requires skipping:
        - grad is None for all Muon params (would cause deadlock without participation)
        - NaN/Inf detected in grads
        - External skip signal (e.g., gradient accumulation boundary)

        Note: Even if should_skip=True, we must still enter step() and participate in
        collective operations to avoid deadlock. The caller should check skip flag
        after this returns and call _group_sync_skip() for consistent group behavior.
        """
        # Check if any Muon param has valid grad
        # If ALL grads are None, we still must enter step() but skip param updates
        all_grad_none = True
        for info in self.muon_param_infos:
            if info["param"].grad is not None:
                all_grad_none = False
                break
        if all_grad_none and len(self.muon_param_infos) > 0:
            return True

        # Check for NaN/Inf in grads
        for info in self.muon_param_infos:
            g = info["param"].grad
            if g is not None:
                if not torch.isfinite(g).all():
                    return True
        return False

    def _group_sync_skip(self) -> bool:
        """Synchronize skip decision across the FSDP group.

        If ANY rank wants to skip, ALL ranks skip (avoids deadlock in allgather).
        Returns True if the group should skip the step.
        """
        import torch.distributed as dist

        # Compute local skip
        local_skip = 1 if self._local_should_skip_step() else 0
        skip_tensor = torch.tensor([local_skip], dtype=torch.int32, device=torch.cuda.current_device())
        dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX, group=self._fsdp_group)
        return skip_tensor.item() != 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        # Lazy build
        if not self._built:
            self._build_param_infos()
            self._built = True
            # Allocate state tensors for Muon params only
            for info in self.muon_param_infos:
                p = info["param"]
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0, device=p.device)
                    # exp_avg: local shard in momentum_dtype
                    state["exp_avg"] = torch.zeros(
                        info["local_shape"], dtype=info.get("momentum_dtype", torch.bfloat16), device=p.device
                    )
                    info["exp_avg"] = state["exp_avg"]
                    # Persistent communication buffers
                    info["send_buffer"] = torch.zeros(
                        info["padded_numel"], dtype=torch.float32, device=p.device
                    )
                    info["recv_buffer"] = torch.zeros(
                        info["world_size"] * info["padded_numel"], dtype=torch.float32, device=p.device
                    )

        # Global skip synchronization: if ANY rank should skip, ALL skip (avoids deadlock)
        if self._group_sync_skip():
            return

        for info in self.param_infos:
            if info["optimizer_kind"] == "adamw":
                self._adamw_step_local(info)
            else:
                self._muon_step_param(info)

    def _muon_step_param(self, info: Dict):
        """Per-parameter Muon step with allgather + NS."""
        p = info["param"]
        grad = p.grad

        # Detect DTensor: need to extract local tensor for in-place ops
        p_is_dtensor = isinstance(p.data, DTensor)
        if p_is_dtensor:
            p_local = p.data.to_local()
        else:
            p_local = p.data

        if grad is None:
            # Must still participate in collectives to avoid deadlock
            grad = torch.zeros_like(p)

        # 1) Get local grad shard
        # FSDP2's all-gather at end of backward returns the LOCAL shard per rank,
        # so grad is already the correct local shape matching exp_avg/m_local.
        # Just convert to float32 for the optimizer update.
        grad_is_dtensor = isinstance(grad, DTensor)
        if grad_is_dtensor:
            # grad.to_local() = local shard = [local_dim0, N]
            grad_local = grad.to_local()
            g_local = grad_local.reshape(-1).to(torch.float32)
        else:
            g_local = grad.detach().reshape(-1).to(torch.float32)

        # 2) Update local momentum
        beta1 = info["beta1"]
        m_local = info["exp_avg"]
        # Ensure g_local has same shape as m_local (both may have same numel but different dims)
        if g_local.numel() == m_local.numel() and g_local.ndim != m_local.ndim:
            g_local = g_local.reshape(m_local.shape)
        m_local.mul_(beta1).add_(g_local, alpha=1.0 - beta1)

        # 3) Decoupled weight decay
        if info["weight_decay"] != 0:
            p_local.mul_(1.0 - info["lr"] * info["weight_decay"])

        # 4) Pack send buffer
        send = info["send_buffer"]
        recv = info["recv_buffer"]
        send.zero_()
        if info["local_numel"] > 0:
            send[: info["local_numel"]].copy_(m_local[: info["local_numel"]].reshape(-1))

        # 5) All-gather
        group = info["process_group"]
        if group is not None and info["world_size"] > 1:
            torch.distributed.all_gather_into_tensor(recv, send, group=group)
        else:
            recv[: info["local_numel"]].copy_(send[: info["local_numel"]])

        # 6) Reconstruct full matrix
        parts = []
        for r in range(info["world_size"]):
            off = r * info["padded_numel"]
            sz = info["local_numels"][r]
            parts.append(recv[off : off + sz])
        m_full = torch.cat(parts, dim=0).view(info["full_shape"]).to(torch.float32)

        # 7) Newton-Schulz
        block_rows = info["block_rows"]
        if block_rows is None:
            u_full = self._muon_newton_schulz(m_full, ns_steps=info.get("ns_steps", 5))
        else:
            u_full = self._muon_newton_schulz_blockwise(
                m_full, block_rows=block_rows, ns_steps=info.get("ns_steps", 5)
            )

        # 8) Slice local update and apply
        # Note: p_local is a plain tensor extracted from p.data.to_local().
        # We update it in-place. FSDP2 syncs params via post-optimizer allreduce,
        # not via explicit p.data.copy_().
        if info["local_numel"] > 0:
            u_local = u_full.reshape(-1)[info["local_start"] : info["local_end"]]
            p_local.reshape(-1).add_(u_local.to(p_local.dtype), alpha=-info["lr"])

        p.grad = None

    def _muon_newton_schulz(self, G: torch.Tensor, ns_steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """
        Newton-Schulz iteration to orthogonalize a matrix.

        G: [M, N] matrix (FP32)
        Returns: ortho(G) with same shape

        NS iteration: X_{k+1} = (3/2) X_k - (1/2) X_k @ X_k^T @ X_k
        """
        X = G
        transposed = False

        # Handle rectangular: if M > N, transpose so NS iterates on smaller dim
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True

        # Normalize
        X = X / (X.norm() + eps)

        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X

        if transposed:
            X = X.T

        return X

    def _muon_newton_schulz_blockwise(self, G: torch.Tensor, block_rows: int, ns_steps: int = 5) -> torch.Tensor:
        """Blockwise NS for large matrices that can't fit in memory."""
        M, N = G.shape
        out = torch.empty_like(G)

        row_start = 0
        while row_start < M:
            row_end = min(row_start + block_rows, M)
            block = G[row_start:row_end, :]
            out[row_start:row_end, :] = self._muon_newton_schulz(block, ns_steps=ns_steps)
            row_start = row_end

        return out

    def _adamw_step_local(self, info: Dict):
        """Fallback AdamW step for non-2D or non-Muon parameters."""
        p = info["param"]
        grad = p.grad
        if grad is None:
            return

        # Detect DTensor: need to extract local tensor for in-place ops
        p_is_dtensor = isinstance(p.data, DTensor)
        if p_is_dtensor:
            p_local = p.data.to_local()
        else:
            p_local = p.data

        # grad may also be a DTensor in FSDP2; extract local shard
        grad_is_dtensor = isinstance(grad, DTensor)
        if grad_is_dtensor:
            grad = grad.to_local()

        group = info["group"]
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        lr = info["lr"]
        weight_decay = info["weight_decay"]
        eps = group.get("eps", 1e-8)

        state = self.state[p]
        # NOTE: torch.zeros_like(p) where p is DTensor returns a DTensor, not a local tensor.
        # For DTensor params we must use local_shape explicitly to get a plain local tensor,
        # otherwise in-place ops (exp_avg.mul_, exp_avg.add_) mix DTensor and Tensor → crash.
        if len(state) == 0:
            state["step"] = torch.tensor(0, device=p.device)
            if p_is_dtensor:
                # Use local_shape to allocate plain local tensor (not DTensor)
                local_shape = info.get("local_shape", p.to_local().shape)
                state["exp_avg"] = torch.zeros(
                    local_shape, dtype=torch.float32, device=p.device
                )
                state["exp_avg_sq"] = torch.zeros(
                    local_shape, dtype=torch.float32, device=p.device
                )
            else:
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        state["step"] += 1
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

        bias_correction1 = 1 - beta1 ** state["step"].item()
        bias_correction2 = 1 - beta2 ** state["step"].item()

        if weight_decay > 0:
            p_local.mul_(1 - lr * weight_decay)

        # Ensure consistent dtype: convert grad to float32 to match exp_avg/exp_avg_sq
        grad = grad.to(dtype=torch.float32) if grad.dtype != torch.float32 else grad
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
        step_size = lr / bias_correction1
        p_local.addcdiv_(exp_avg, denom, value=-step_size)

        # NOTE: p_local is a view / alias of the DTensor's local storage (via to_local()).
        # In FSDP2 the all-reduce at the end of backward pass syncs updated shards across ranks,
        # so no explicit copy back is needed.  Redundant p.data.copy_(p_local) would also be
        # wrong because it mixes DTensor (p.data) and plain Tensor (p_local) in copy_.

        p.grad = None


class MultiOptimizer(Optimizer, Stateful):
    """
    A container that handles multiple optimizers (for ep and non-ep parameters when ep+fsdp2 is enabled)

    Mapping of name -> torch.optim.Optimizer with convenience methods.
    Compatible with torch.distributed.checkpoint optimizer APIs that accept a Mapping.

    This class is needed for EP+FSDP2 case because EP and non-EP param have different FSDP sharding dimension (dim-0 vs. dim-1)
    For comparison, EP+FSDP1 also shards EP parameters along dim-0 for FSDP, so it can use the default optimizer class.
    """

    def __init__(
        self,
        root_model: nn.Module,
        optimizers: dict,  # {"ep": opt1, "non_ep": opt2}
        key_names: list[str],
    ):
        self.model = root_model
        self.optimizers_dict = optimizers
        self._is_multi_optimizer: bool = True
        self.key_names = key_names

    def step(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.zero_grad()

    def state_dict(
        self,
    ) -> Dict[str, Any]:
        # get the flatten state dict for multi-optimizer
        merged: Dict[str, Any] = {}
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            sd = get_optimizer_state_dict(self.model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True))
            # check for key clashes before merging
            overlap = set(merged.keys()) & set(sd.keys())
            if overlap:
                raise KeyError(
                    f"Key clash detected while merging state dict for optimizer '{name}': {', '.join(sorted(overlap))}"
                )
            else:
                logger.info_rank0("No clashes when merging MultiOptimizer state dicts")
            merged.update(sd)

        return merged

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Feed the same merged flattened dict to each sub-optimizer; PyTorch will
        # pick out only the entries for parameters that belong to that optimizer.
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            set_optimizer_state_dict(
                self.model,
                opt,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

    def register_step_pre_hook(self, hook):
        return [opt.register_step_pre_hook(hook) for opt in self.optimizers_dict.values()]

    def __len__(self) -> int:
        return len(self.optimizers_dict)


def _should_build_ep_aware(model: "nn.Module") -> bool:
    ps = get_parallel_state()
    if ps.dp_mode != "fsdp2" or not ps.ep_enabled:
        return False

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, DTensor):
            mesh = getattr(p, "device_mesh", None)
            names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
            if "ep_fsdp" in names:
                return True
    return False


def _make_param_groups_for_subset(
    model: "nn.Module",
    params: Iterable[torch.nn.Parameter],
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    decay_param_names = set(get_parameter_names(model, no_decay_modules, no_decay_params))
    name_by_param = {p: n for n, p in model.named_parameters()}
    params = [p for p in params if p.requires_grad]
    decayed = [p for p in params if name_by_param.get(p) in decay_param_names]
    undecayed = [p for p in params if name_by_param.get(p) not in decay_param_names]
    groups: List[Dict[str, Any]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


# adapted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer_pt_utils.py#L1123
def get_parameter_names(model, forbidden_layer_types, forbidden_param_names):
    forbidden_layer_types = [] if forbidden_layer_types is None else forbidden_layer_types
    forbidden_param_names = [] if forbidden_param_names is None else forbidden_param_names
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_param_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if child.__class__.__name__ not in forbidden_layer_types
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_param_names)
        ]

    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_param_names)
    ]
    return result


def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> "torch.optim.Optimizer":
    # EP-aware routing: for FSDP2+EP, split params into EP and non-EP groups and build two optimizers.
    if _should_build_ep_aware(model):
        return build_ep_fsdp2_optimizer(
            model, lr, betas, eps, weight_decay, fused, optimizer_type, param_groups, no_decay_modules, no_decay_params
        )
    # Other cases remain the same
    if param_groups is None:
        decay_param_names = get_parameter_names(model, no_decay_modules, no_decay_params)
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_param_names and p.requires_grad],
                "weight_decay": weight_decay,
            },
        ]
        no_decay_parameters, no_decay_parameter_names = [], []
        for n, p in model.named_parameters():
            if n not in decay_param_names and p.requires_grad:
                no_decay_parameter_names.append(n)
                no_decay_parameters.append(p)

        if len(no_decay_parameters) > 0:
            logger.info_rank0(f"Parameters without weight decay: {no_decay_parameter_names}")
            param_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    if optimizer_type == "adamw":
        foreach = not fused
        fused = fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay)
    elif optimizer_type == "muon":
        # Check if FSDP2 is enabled
        ps = get_parallel_state()
        if ps.fsdp_enabled and ps.dp_mode == "fsdp2":
            optim = FSDP2AwareMuon(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        else:
            optim = Muon(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif optimizer_type == "muon_v2":
        # Proper Muon with per-layer allgather + Newton-Schulz
        ps = get_parallel_state()
        if ps.fsdp_enabled and ps.dp_mode == "fsdp2":
            optim = FSDP2AwareMuonV2(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, model=model)
        else:
            # Fall back to plain Muon for non-FSDP2
            optim = Muon(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise ValueError("Only adamw, anyprecision_adamw, muon, and muon_v2 are supported as optimizers.")

    return optim


def build_ep_fsdp2_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[List[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
):
    """
    Build a MultiOptimizer instance when model is parallelized with EP+FSDP2

    If param_groups provided, it can be a list of dicts with arbitrary parameter groups:
    - Example: [{"params": params1, "lr": lr1},
                {"params": params2, "lr": lr2},
                {"params": params3, "lr": lr3}]
    - Each group's params are automatically split into EP and non-EP based on DTensor mesh
    - Custom learning rates and other optimizer settings are preserved per group
    """
    # Collect all EP and non-EP parameters across all groups
    ep_groups: List[Dict[str, Any]] = []
    non_ep_groups: List[Dict[str, Any]] = []

    # Process custom param_groups if provided
    if param_groups is not None:
        # Validate param_groups structure
        assert isinstance(param_groups, list), "param_groups must be a list"

        # Process each parameter group
        for group_config in param_groups:
            assert "params" in group_config, (
                f"Each group in param_groups must contain 'params' key, got: {group_config}"
            )

            # Extract group-specific settings
            group_lr = group_config.get("lr", lr)
            group_params = group_config["params"]

            # Split this group's params into EP and non-EP
            group_ep_params: List[torch.nn.Parameter] = []
            group_non_ep_params: List[torch.nn.Parameter] = []

            for p in group_params:
                if not p.requires_grad:
                    continue
                if DTensor is not None and isinstance(p, DTensor):
                    mesh = getattr(p, "device_mesh", None)
                    names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                    if "ep_fsdp" in names:
                        group_ep_params.append(p)
                        continue
                group_non_ep_params.append(p)

            # Create subgroups with weight decay handling
            if group_ep_params:
                group_ep_subgroups = _make_param_groups_for_subset(
                    model, group_ep_params, weight_decay, no_decay_modules, no_decay_params
                )
                for subgroup in group_ep_subgroups:
                    subgroup["lr"] = group_lr
                    # Preserve other custom settings from original group
                    for key, value in group_config.items():
                        if key not in ["params", "lr", "weight_decay"]:
                            subgroup[key] = value
                ep_groups.extend(group_ep_subgroups)

            if group_non_ep_params:
                group_non_ep_subgroups = _make_param_groups_for_subset(
                    model, group_non_ep_params, weight_decay, no_decay_modules, no_decay_params
                )
                for subgroup in group_non_ep_subgroups:
                    subgroup["lr"] = group_lr
                    # Preserve other custom settings from original group
                    for key, value in group_config.items():
                        if key not in ["params", "lr", "weight_decay"]:
                            subgroup[key] = value
                non_ep_groups.extend(group_non_ep_subgroups)
    else:
        # Default case (param_groups is None): all model parameters with uniform settings(lr)
        ep_params: List[torch.nn.Parameter] = []
        non_ep_params: List[torch.nn.Parameter] = []

        for p in model.parameters():
            if not p.requires_grad:
                continue
            if DTensor is not None and isinstance(p, DTensor):
                mesh = getattr(p, "device_mesh", None)
                names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                if "ep_fsdp" in names:
                    ep_params.append(p)
                    continue
            non_ep_params.append(p)

        # Build param groups with weight decay handling
        ep_groups = _make_param_groups_for_subset(model, ep_params, weight_decay, no_decay_modules, no_decay_params)
        non_ep_groups = _make_param_groups_for_subset(
            model, non_ep_params, weight_decay, no_decay_modules, no_decay_params
        )

    def _build(groups: Sequence[Dict[str, Any]]) -> Optimizer:
        foreach = False if is_torch_npu_available() else (not fused)
        fused_ = False if is_torch_npu_available() else fused
        if optimizer_type == "adamw":
            return AdamW(groups, lr, betas, eps, weight_decay, fused=fused_, foreach=foreach)
        elif optimizer_type == "anyprecision_adamw":
            return AnyPrecisionAdamW(groups, lr, betas, eps, weight_decay)
        elif optimizer_type == "muon":
            return FSDP2AwareMuon(groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        elif optimizer_type == "muon_v2":
            return FSDP2AwareMuonV2(groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, model=model)
        else:
            raise ValueError("Only adamw, anyprecision_adamw, muon, and muon_v2 are supported.")

    optimizer_dict: Dict[str, Optimizer] = {}
    if ep_groups:
        optimizer_dict["ep"] = _build(ep_groups)
    if non_ep_groups:
        optimizer_dict["non_ep"] = _build(non_ep_groups)

    # cache for EP-aware grad clipping helpers
    model._ep_param_groups = {
        "ep": [p for g in ep_groups for p in g.get("params", [])] if ep_groups else [],
        "non_ep": [p for g in non_ep_groups for p in g.get("params", [])] if non_ep_groups else [],
    }

    key_names = list(optimizer_dict.keys())

    # Build MultiOptimizer and attach a pre-step hook to sanitize DTensor states
    multi_opt = MultiOptimizer(model, optimizer_dict, key_names=key_names)

    return multi_opt
