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

import torch
import torch.distributed as dist

from ..parallel_state import get_parallel_state
from ...utils import logging

logger = logging.get_logger(__name__)

def _get_pp_group():
    ps = get_parallel_state()
    return ps.pp_mesh.get_group()

def _get_next_rank():
    ps = get_parallel_state()
    rank = ps.pp_rank
    world_size = ps.pp_size
    return (rank + 1) % world_size

def _get_prev_rank():
    ps = get_parallel_state()
    rank = ps.pp_rank
    world_size = ps.pp_size
    return (rank - 1 + world_size) % world_size

def send_forward(output_tensor: torch.Tensor) -> None:
    """Send tensor to the next stage in the forward pass."""
    ps = get_parallel_state()
    if ps.is_last_pp_stage:
        return
    
    next_rank = _get_next_rank()
    group = _get_pp_group()
    
    output_tensor = output_tensor.contiguous()
    dist.send(output_tensor, dst=next_rank, group=group)

def recv_forward(tensor_shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Receive tensor from the previous stage in the forward pass."""
    ps = get_parallel_state()
    if ps.is_first_pp_stage:
        return None
    
    prev_rank = _get_prev_rank()
    group = _get_pp_group()
    
    recv_tensor = torch.empty(tensor_shape, dtype=dtype, device=device)
    dist.recv(recv_tensor, src=prev_rank, group=group)
    return recv_tensor

def send_backward(input_tensor_grad: torch.Tensor) -> None:
    """
    Send gradient to the previous stage in the backward pass.
    This corresponds to the gradient of the input tensor we received in forward.
    """
    ps = get_parallel_state()
    if ps.is_first_pp_stage:
        return
    
    prev_rank = _get_prev_rank()
    group = _get_pp_group()
    
    input_tensor_grad = input_tensor_grad.contiguous()
    dist.send(input_tensor_grad, dst=prev_rank, group=group)

def recv_backward(tensor_shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Receive gradient from the next stage in the backward pass.
    This corresponds to the gradient of the output tensor we sent in forward.
    """
    ps = get_parallel_state()
    if ps.is_last_pp_stage:
        return None
    
    next_rank = _get_next_rank()
    group = _get_pp_group()
    
    # Gradient shape should match the output shape (which usually matches input shape of next stage)
    recv_tensor = torch.empty(tensor_shape, dtype=dtype, device=device)
    dist.recv(recv_tensor, src=next_rank, group=group)
    return recv_tensor
