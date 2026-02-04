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

"""
Ring Attention integration for Context Parallelism (CP).

This module integrates ring-flash-attn for efficient long-context training
with context parallelism. It supports both varlen (packing) and batch APIs.

Usage:
    1. Initialize CP group via init_parallel_state with cp_size > 1
    2. Call substitute_hf_ring_attn() to replace HF flash attention
    3. Before each forward, call update_ring_attn_params() with cu_seqlens
"""

from typing import Optional, Tuple, TYPE_CHECKING
import torch
from torch import distributed as dist

from ...utils import logging

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.get_logger(__name__)

# Global state for ring attention
_RING_ATTN_GROUP: Optional["ProcessGroup"] = None
_RING_ATTN_CU_SEQLENS: Optional[torch.Tensor] = None
_RING_ATTN_SUBSTITUTED: bool = False

# Check if ring-flash-attn is available
try:
    from ring_flash_attn import (
        substitute_hf_flash_attn,
        update_ring_flash_attn_params,
        zigzag_ring_flash_attn_func,
        zigzag_ring_flash_attn_varlen_func,
        llama3_flash_attn_varlen_func,
    )
    RING_FLASH_ATTN_AVAILABLE = True
except ImportError:
    RING_FLASH_ATTN_AVAILABLE = False
    substitute_hf_flash_attn = None
    update_ring_flash_attn_params = None
    zigzag_ring_flash_attn_func = None
    zigzag_ring_flash_attn_varlen_func = None
    llama3_flash_attn_varlen_func = None


def is_ring_flash_attn_available() -> bool:
    """Check if ring-flash-attn is installed."""
    return RING_FLASH_ATTN_AVAILABLE


def init_ring_attention(cp_group: "ProcessGroup") -> None:
    """
    Initialize ring attention with the given context parallel group.
    
    Args:
        cp_group: The process group for context parallelism
    """
    global _RING_ATTN_GROUP
    
    if not RING_FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "ring-flash-attn is not installed. "
            "Please install it with: pip install ring-flash-attn"
        )
    
    _RING_ATTN_GROUP = cp_group
    logger.info_rank0(
        f"Ring attention initialized with group size {dist.get_world_size(cp_group)}"
    )


def get_ring_attention_group() -> Optional["ProcessGroup"]:
    """Get the ring attention process group."""
    return _RING_ATTN_GROUP


def substitute_hf_ring_attn(
    heads_k_stride: int = 1,
    use_llama3_style: bool = True,
) -> None:
    """
    Substitute HuggingFace flash attention with ring attention.
    
    This should be called after model loading but before training.
    
    Args:
        heads_k_stride: Stride for K heads (for GQA models)
        use_llama3_style: Use llama3-style ring attention (recommended for varlen)
    """
    global _RING_ATTN_SUBSTITUTED
    
    if not RING_FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "ring-flash-attn is not installed. "
            "Please install it with: pip install ring-flash-attn"
        )
    
    if _RING_ATTN_GROUP is None:
        raise RuntimeError(
            "Ring attention group not initialized. "
            "Call init_ring_attention() first."
        )
    
    if _RING_ATTN_SUBSTITUTED:
        logger.warning_rank0("Ring attention already substituted, skipping.")
        return
    
    substitute_hf_flash_attn(_RING_ATTN_GROUP, heads_k_stride=heads_k_stride)
    _RING_ATTN_SUBSTITUTED = True
    
    logger.info_rank0(
        f"HuggingFace flash attention substituted with ring attention "
        f"(heads_k_stride={heads_k_stride})"
    )


def update_ring_attn_cu_seqlens(cu_seqlens: torch.Tensor) -> None:
    """
    Update the cu_seqlens for ring attention.
    
    This should be called before each forward pass when using varlen API.
    
    Args:
        cu_seqlens: Cumulative sequence lengths tensor [num_seqs + 1]
    """
    global _RING_ATTN_CU_SEQLENS
    
    if not RING_FLASH_ATTN_AVAILABLE:
        return
    
    if _RING_ATTN_GROUP is None:
        return
    
    _RING_ATTN_CU_SEQLENS = cu_seqlens
    update_ring_flash_attn_params(cu_seqlens, _RING_ATTN_GROUP)


def get_ring_attn_cu_seqlens() -> Optional[torch.Tensor]:
    """Get the current cu_seqlens for ring attention."""
    return _RING_ATTN_CU_SEQLENS


def ring_flash_attn_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    use_varlen: bool = False,
    use_zigzag: bool = True,
) -> torch.Tensor:
    """
    Ring attention forward pass.
    
    Args:
        query: Query tensor [batch, seqlen, num_heads, head_dim] or [total_q, num_heads, head_dim]
        key: Key tensor [batch, seqlen, num_kv_heads, head_dim] or [total_k, num_kv_heads, head_dim]
        value: Value tensor [batch, seqlen, num_kv_heads, head_dim] or [total_k, num_kv_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries (varlen only)
        cu_seqlens_k: Cumulative sequence lengths for keys (varlen only)
        max_seqlen_q: Maximum query sequence length (varlen only)
        max_seqlen_k: Maximum key sequence length (varlen only)
        dropout_p: Dropout probability
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim))
        causal: Whether to use causal attention
        use_varlen: Whether to use varlen API
        use_zigzag: Whether to use zigzag ring attention (more balanced)
        
    Returns:
        Output tensor with same shape as query
    """
    if not RING_FLASH_ATTN_AVAILABLE:
        raise ImportError("ring-flash-attn is not installed")
    
    if _RING_ATTN_GROUP is None:
        raise RuntimeError("Ring attention group not initialized")
    
    if use_varlen:
        if cu_seqlens_q is None or max_seqlen_q is None:
            raise ValueError("cu_seqlens_q and max_seqlen_q required for varlen API")
        
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        if max_seqlen_k is None:
            max_seqlen_k = max_seqlen_q
        
        if use_zigzag:
            output = zigzag_ring_flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=_RING_ATTN_GROUP,
            )
        else:
            output = llama3_flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=_RING_ATTN_GROUP,
            )
    else:
        if use_zigzag:
            output = zigzag_ring_flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=_RING_ATTN_GROUP,
            )
        else:
            # Fallback to zigzag for batch API
            output = zigzag_ring_flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=_RING_ATTN_GROUP,
            )
    
    return output


def prepare_inputs_for_ring_attn(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    cp_group: "ProcessGroup",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepare inputs for ring attention by chunking across CP ranks.
    
    Args:
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        position_ids: Position IDs [batch, seq_len]
        cp_group: Context parallel process group
        
    Returns:
        Tuple of (chunked_input_ids, chunked_attention_mask, chunked_position_ids)
    """
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    
    batch_size, seq_len = input_ids.shape
    
    # Pad sequence length to be divisible by cp_size
    if seq_len % cp_size != 0:
        pad_len = cp_size - (seq_len % cp_size)
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
        if attention_mask is not None:
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
        if position_ids is not None:
            # Extend position_ids
            last_pos = position_ids[:, -1:] + 1
            pad_positions = last_pos + torch.arange(pad_len, device=position_ids.device)
            position_ids = torch.cat([position_ids, pad_positions.expand(batch_size, -1)], dim=1)
        seq_len = input_ids.shape[1]
    
    # Chunk across CP dimension
    chunk_size = seq_len // cp_size
    start_idx = cp_rank * chunk_size
    end_idx = start_idx + chunk_size
    
    chunked_input_ids = input_ids[:, start_idx:end_idx].contiguous()
    chunked_attention_mask = attention_mask[:, start_idx:end_idx].contiguous() if attention_mask is not None else None
    chunked_position_ids = position_ids[:, start_idx:end_idx].contiguous() if position_ids is not None else None
    
    return chunked_input_ids, chunked_attention_mask, chunked_position_ids


def gather_outputs_from_ring_attn(
    output: torch.Tensor,
    cp_group: "ProcessGroup",
    dim: int = 1,
) -> torch.Tensor:
    """
    Gather outputs from all CP ranks.
    
    Args:
        output: Local output tensor
        cp_group: Context parallel process group
        dim: Dimension to gather along
        
    Returns:
        Gathered output tensor
    """
    cp_size = dist.get_world_size(cp_group)
    
    if cp_size == 1:
        return output
    
    # All-gather outputs
    gathered_list = [torch.zeros_like(output) for _ in range(cp_size)]
    dist.all_gather(gathered_list, output, group=cp_group)
    
    return torch.cat(gathered_list, dim=dim)
