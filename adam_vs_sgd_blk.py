import random
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def init_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

init_seed(0)

def _compute_default_rope_parameters(
    rope_theta: float,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
  
    base = rope_theta
    partial_rotary_factor = 1.0
    head_dim = 896//14
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        self.rope_kwargs = {
            "rope_type": rope_type,
            "factor": scaling_factor,
            "dim": dim,
            "base": base,
            "max_position_embeddings": max_position_embeddings,
        }
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.rope_init_fn = _compute_default_rope_parameters
        inv_freq, self.attention_scaling = self.rope_init_fn( rope_theta=1000000.0,device=device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            freqs = torch.matmul(inv_freq_expanded.float(), position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class MLP(nn.Module):
    def __init__(self, hidden_size=896, intermediate_size=4864, act_fn=nn.SiLU):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, hidden_size=9, num_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_size = 896
        self.num_heads = 14
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = 32768
        self.rope_theta = 1000000.0
        self.is_causal = True
        self.attention_dropout = 0.0

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding()

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()


        # # with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
        # # with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        # with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        # # with torch.nn.attention.sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        #     attn_output = scaled_dot_product_attention(
        #         query_states,
        #         key_states,
        #         value_states,
        #         attn_mask=None,
        #         dropout_p=self.attention_dropout if self.training else 0.0,
        #         is_causal=True,
        #     )


        # flash attention https://github.com/Dao-AILab/flash-attention
        # attn_output = flash_attn_func(
        #     query_states,
        #     key_states,
        #     value_states,
        #     dropout_p=0.0,
        #     causal=True,
        #     deterministic=True,
        # )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )

        # attn_weight = torch.matmul(query_states, key_states.transpose(-2, -1))
        # attn_weight = attn_weight / (float(self.head_dim) ** 0.5)
        # attn_weight = F.softmax(attn_weight, dim=-1)
        # attn_output = torch.matmul(attn_weight, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, 

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size=896, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, hidden_size = 896, ):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = Attention()
        self.mlp = MLP()
        self.input_layernorm = Qwen2RMSNorm(896, eps=1e-06)
        self.post_attention_layernorm = Qwen2RMSNorm(896, eps=1e-06)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) :

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs[0]

data = torch.randn(16, 2048, 896).cuda().to(torch.bfloat16)

label = torch.randn(16, 2048, 896).cuda().to(torch.bfloat16)

model = Qwen2DecoderLayer(896).cuda().to(torch.bfloat16)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

position_ids = torch.arange(data.shape[1], dtype=torch.long, device=data.device).unsqueeze(0).cuda().to(torch.bfloat16)

import time
start_time = time.time()
torch.cuda.synchronize()
for i in range(100):
    optimizer.zero_grad()
    output = model(data,position_ids=position_ids)
    loss = F.mse_loss(output, label)
    loss.backward()
    optimizer.step()
    # print(loss.item())
    if i % 10 == 0:
        print(model.self_attn.q_proj.weight.grad.mean().item())

torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken: {end_time - start_time}")