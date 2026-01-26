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

import torch
import torch.nn as nn

from ..parallel_state import get_parallel_state
from .p2p import recv_backward, recv_forward, send_backward, send_forward

class PipelineStage(nn.Module):
    def __init__(self, local_model: nn.Module, input_shape=None):
        super().__init__()
        self.local_model = local_model
        self.input_shape = input_shape
        self.ps = get_parallel_state()

        # State for backward pass (FIFO for 1F1B)
        self._input_queue = deque()
        self._output_queue = deque()
        self._output_shape_queue = deque()
        
    def forward(self, x=None, *args, **kwargs):
        # 1. Receive input if not first stage
        if not self.ps.is_first_pp_stage:
            if self.input_shape is None:
                raise ValueError("pp_input_shape must be provided for non-first pipeline stages.")
            # Infer dtype/device from local model parameters when possible
            param = next(self.local_model.parameters(), None)
            dtype = param.dtype if param is not None else torch.bfloat16
            device = param.device if param is not None else torch.device(f"cuda:{self.ps.local_rank}")
            x = recv_forward(self.input_shape, dtype, device)
            
            # Enable grad for input so we can compute gradient w.r.t it
            x.requires_grad_(True)
            self._input_queue.append(x)
        else:
            # Inputs from dataloader, first stage doesn't need to send grad back.
            self._input_queue.append(None)
            
        # 2. Local Forward
        output = self.local_model(x, *args, **kwargs)
        self._output_queue.append(output)
        
        # 3. Send output if not last stage
        if not self.ps.is_last_pp_stage:
            # Handle tuple output (hidden_states, ...)
            tensor_to_send = output
            if isinstance(output, tuple):
                tensor_to_send = output[0]

            self._output_shape_queue.append(tensor_to_send.shape)
            send_forward(tensor_to_send)
        else:
            # Last stage doesn't send forward activations
            self._output_shape_queue.append(None)
                
        return output

    def backward_step(self, loss=None):
        """
        Execute backward pass for this stage.
        Args:
            loss: scalar loss (only valid for last stage).
        """
        if not self._output_queue:
            raise RuntimeError("PipelineStage backward_step called with no saved forward activations.")

        output = self._output_queue.popleft()
        input_tensor = self._input_queue.popleft()
        output_shape = self._output_shape_queue.popleft()

        # 1. Get Gradient for Output
        if self.ps.is_last_pp_stage:
            if loss is None:
                raise ValueError("Loss must be provided for the last pipeline stage.")
            loss.backward()
        else:
            dtype = output.dtype if isinstance(output, torch.Tensor) else torch.bfloat16
            device = torch.device(f"cuda:{self.ps.local_rank}")
            output_grad = recv_backward(output_shape, dtype, device)

            target_tensor = output
            if isinstance(target_tensor, tuple):
                target_tensor = target_tensor[0]

            torch.autograd.backward(target_tensor, output_grad)

        # 2. Send Gradient for Input (if needed)
        if not self.ps.is_first_pp_stage:
            if input_tensor is not None and input_tensor.grad is not None:
                send_backward(input_tensor.grad)
