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

import copy
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from ..parallel_state import get_parallel_state
from ...utils import logging

logger = logging.get_logger(__name__)

class LlamaPipelineStage(nn.Module):
    """
    A local pipeline stage for Llama-like models.
    """
    def __init__(self, model, pp_rank, pp_size):
        super().__init__()
        self.config = model.config
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        
        # 1. Distribute Layers
        num_layers = self.config.num_hidden_layers
        layers_per_rank = num_layers // pp_size
        start_layer = pp_rank * layers_per_rank
        end_layer = (pp_rank + 1) * layers_per_rank if pp_rank != pp_size - 1 else num_layers
        
        self.local_layers = nn.ModuleList()
        # We assume model.model.layers is the ModuleList
        if hasattr(model, "model") and hasattr(model.model, "layers"):
             full_layers = model.model.layers
        elif hasattr(model, "layers"): # Fallback
             full_layers = model.layers
        else:
            raise ValueError("Could not find layers in model")
            
        for i in range(start_layer, end_layer):
            self.local_layers.append(full_layers[i])
            
        # 2. Shared Components (Rotary Emb)
        # Every stage needs rotary embeddings to generate pos_embeds for its layers
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            self.rotary_emb = model.model.rotary_emb
        elif hasattr(model, "rotary_emb"):
            self.rotary_emb = model.rotary_emb
        else:
             self.rotary_emb = None # Might be inside layers or not needed explicitly

        # 3. First Stage Components
        if self.pp_rank == 0:
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                self.embed_tokens = model.model.embed_tokens
            else:
                self.embed_tokens = model.embed_tokens
        else:
            self.embed_tokens = None
            
        # 4. Last Stage Components
        if self.pp_rank == self.pp_size - 1:
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                self.norm = model.model.norm
            else:
                self.norm = model.norm
            
            if hasattr(model, "lm_head"):
                self.lm_head = model.lm_head
            else:
                # Some custom models might not have lm_head or it's named differently
                self.lm_head = None
        else:
            self.norm = None
            self.lm_head = None
            
        # Clear reference to full model to allow GC (optional, but good practice)
        # However, we must be careful not to delete shared weights if they are references
        
    def forward(self, hidden_states, input_ids=None, position_ids=None, attention_mask=None, **kwargs):
        # 1. Embeddings (Rank 0)
        if self.pp_rank == 0:
            if self.embed_tokens is not None and input_ids is not None:
                hidden_states = self.embed_tokens(input_ids)
        
        # 2. Rotary Embeddings
        # We need to generate position_embeddings. 
        # Usually LlamaRotaryEmbedding takes (value, position_ids)
        # value is used to determine device/dtype and seq_len
        if self.rotary_emb is not None:
            if position_ids is None:
                seq_len = None
                if input_ids is not None:
                    seq_len = input_ids.shape[-1]
                    device = input_ids.device
                elif hidden_states is not None:
                    seq_len = hidden_states.shape[1]
                    device = hidden_states.device
                else:
                    raise ValueError("Cannot infer position_ids without input_ids or hidden_states.")
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        # 3. Layers
        for layer in self.local_layers:
            # LlamaDecoderLayer forward signature usually:
            # (hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None)
            # We pass what we have.
            
            # Note: We assume causal_mask/attention_mask is already prepared and passed correctly.
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs
            )
            hidden_states = layer_outputs[0]

        # 4. Final Norm and Head (Last Rank)
        if self.pp_rank == self.pp_size - 1:
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is not None:
                hidden_states = self.lm_head(hidden_states)
                
        return hidden_states


def partition_model(model: nn.Module) -> nn.Module:
    """
    Partition a Llama-like model into a local stage module.
    """
    ps = get_parallel_state()
    if not ps.pp_enabled:
        return model
        
    logger.info_rank0(f"Partitioning model for Pipeline Parallelism: Rank {ps.pp_rank}/{ps.pp_size}")
    
    # Currently only supports Llama-structure models
    # We create a new module that only contains the local layers/components
    local_stage = LlamaPipelineStage(model, ps.pp_rank, ps.pp_size)
    
    return local_stage
