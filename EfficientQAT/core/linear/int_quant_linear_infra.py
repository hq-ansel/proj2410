from typing import List, Dict, Callable, Union, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantizer import QuantConfig
from .int_quant_linear import IntQuantLinear

@torch.jit.script
def _fused_quant_dequant_impl(weight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int, n_bits: int) -> torch.Tensor:
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    
    # Reshape
    w_reshaped = weight.view(out_features, -1, group_size)
    
    # Ensure scales/zeros match the 3D structure [Out, Groups, 1]
    # Scales and qzeros are stored as [Out, Groups]
    s = scales.view(out_features, -1, 1)
    # Numerical Alignment: Round zero point to match integer packing expectations
    z = qzeros.round().view(out_features, -1, 1)
    
    scale_max = float((1 << n_bits) - 1)
    
    # Quantize with STE: round(w/s) + z
    x = w_reshaped / s
    x_round = x.round()
    x_ste = (x_round - x).detach() + x
    
    w_int_raw = x_ste + z
    w_int = w_int_raw.clamp(0.0, scale_max)
    w_int = (w_int - w_int_raw).detach() + w_int_raw
    
    # Dequantize
    w_deq = (w_int - z) * s
    
    return w_deq.view(out_features, in_features)

@torch.jit.script
def _fused_quant_pack_impl(weight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int, n_bits: int) -> torch.Tensor:
    out_features = weight.shape[0]
    w_reshaped = weight.view(out_features, -1, group_size)
    s = scales.view(out_features, -1, 1)
    z = qzeros.round().view(out_features, -1, 1) # Round here for consistent packing
    
    scale_max = float((1 << n_bits) - 1)
    x = w_reshaped / s
    x_round = x.round()
    
    w_int = (x_round + z).clamp(0.0, scale_max)
    return w_int.to(torch.int32)

@torch.jit.script
def _fused_backward_reduce_impl(grad_W_deq: torch.Tensor, w_int: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # w_int is int32, need float
    w_int_f = w_int.to(grad_W_deq.dtype)
    s = scales.view(grad_W_deq.shape[0], -1, 1)
    z = qzeros.round().view(grad_W_deq.shape[0], -1, 1)
    
    term_s = w_int_f - z
    term_z = -s
    
    grad_scales = (grad_W_deq * term_s).sum(dim=-1)
    grad_qzeros = (grad_W_deq * term_z).sum(dim=-1)
    
    return grad_scales, grad_qzeros

class IntQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, scales, qzeros, bias, g_idx, n_bits, group_size, kernel_backend):
        # 1. Quantize Weight (Float -> Int)
        out_features, in_features = weight.shape
        
        # Use fused quantize with rounded zero points
        w_int = _fused_quant_pack_impl(weight, scales, qzeros, group_size, n_bits)
        
        # Pack Contiguously
        w_int_flat = w_int.view(out_features, in_features)
        qweight = IntQuantLinearInfra.pack_int_data(w_int_flat, n_bits)
        
        ctx.save_for_backward(input, weight, scales, qzeros, w_int, g_idx)
        ctx.n_bits = n_bits
        ctx.group_size = group_size
        ctx.kernel_backend = kernel_backend
        
        # We pass rounded qzeros to kernel for 100% agreement
        output = kernel_backend(
            input, qweight, scales, qzeros.round(), bias=bias, g_idx=g_idx, 
            n_bits=n_bits, group_size=group_size
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, scales, qzeros, w_int, g_idx = ctx.saved_tensors
        n_bits = ctx.n_bits
        group_size = ctx.group_size
        kernel_backend = ctx.kernel_backend
        
        grad_input = None
        grad_weight = None
        grad_scales = None
        grad_qzeros = None
        grad_bias = None
        
        out_features, in_features = weight.shape

        # 1. Gradient w.r.t Input
        if ctx.needs_input_grad[0]:
            w_int_flat = w_int.view(out_features, in_features)
            qweight = IntQuantLinearInfra.pack_int_data(w_int_flat, n_bits)
            
            if hasattr(kernel_backend, 'backward'):
                 grad_input = kernel_backend.backward(
                     grad_output, qweight, scales, qzeros.round(), g_idx=g_idx, 
                     n_bits=n_bits, group_size=group_size
                 )
            else:
                s = scales.view(out_features, -1, 1)
                z = qzeros.round().view(out_features, -1, 1)
                w_deq = (w_int - z) * s
                w_deq = w_deq.view(out_features, in_features)
                grad_input = torch.matmul(grad_output, w_deq)

        # 2. Gradient w.r.t Weight, Scales, Zeros
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_W_deq = torch.matmul(grad_output_2d.t(), input_2d)
        grad_W_deq = grad_W_deq.view(out_features, -1, group_size)
        
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
             g_s, g_z = _fused_backward_reduce_impl(grad_W_deq, w_int, scales, qzeros)
             if ctx.needs_input_grad[2]:
                 grad_scales = g_s.reshape(scales.shape)
             if ctx.needs_input_grad[3]:
                 grad_qzeros = g_z.reshape(qzeros.shape)
            
        if ctx.needs_input_grad[1]:
            grad_weight = grad_W_deq.view(weight.shape)

        if ctx.needs_input_grad[4] and grad_output is not None:
            grad_bias = grad_output_2d.sum(0)

        return grad_input, grad_weight, grad_scales, grad_qzeros, grad_bias, None, None, None, None

class IntQuantLinearInfra(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prefix: str = "",
        config: QuantConfig | None = None,
        kernel_backend: Optional[Any] = None,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prefix = prefix
        self.config = config if config is not None else QuantConfig()
        
        self.group_size = self.config.group_size
        self.n_bits = self.config.n_bits
        num_groups = in_features // self.group_size if self.group_size > 0 else 1
        
        self.register_parameter('weight', nn.Parameter(torch.Tensor(out_features, in_features)))
        self.register_parameter('scales', nn.Parameter(torch.ones(out_features, num_groups)))
        self.register_parameter('qzeros', nn.Parameter(torch.zeros(out_features, num_groups)))
        self.register_buffer('g_idx', None)
        
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)

        self.kernel_backend = kernel_backend
        self.pack_factor = 32 // self.n_bits

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.kernel_backend is not None:
            return IntQuantFunction.apply(
                input, self.weight, self.scales, self.qzeros, self.bias,
                self.g_idx, self.n_bits, self.group_size, self.kernel_backend
            )
        return self._reference_forward_unpacked(input)

    def _reference_forward_unpacked(self, input: torch.Tensor) -> torch.Tensor:
        w_deq = _fused_quant_dequant_impl(self.weight, self.scales, self.qzeros, self.group_size, self.n_bits)
        return F.linear(input, w_deq, self.bias)

    @staticmethod
    def pack_int_data(data: torch.Tensor, n_bits: int) -> torch.Tensor:
        if n_bits >= 32:
            return data.to(torch.int32).contiguous()
        out_features, in_features = data.shape
        pack_factor = 32 // n_bits
        mask = (1 << n_bits) - 1
        data_int = data.to(torch.int32) & mask
        data_reshaped = data_int.view(out_features, -1, pack_factor)
        shifts = torch.arange(0, 32, n_bits, device=data.device, dtype=torch.int32)
        packed = torch.zeros((out_features, in_features // pack_factor), dtype=torch.int32, device=data.device)
        data_shifted = (data_reshaped << shifts)
        for i in range(pack_factor):
            packed |= data_shifted[..., i]
        return packed.contiguous()

    @staticmethod
    def unpack_int_data(packed: torch.Tensor, n_bits: int, original_in_features: int) -> torch.Tensor:
        if n_bits >= 32:
            return packed
        pack_factor = 32 // n_bits
        mask = (1 << n_bits) - 1
        unpacked_list = []
        for i in range(pack_factor):
            unpacked_list.append((packed >> (n_bits * i)) & mask)
        w_int = torch.stack(unpacked_list, dim=-1).flatten(-2)
        return w_int

    @classmethod
    @torch.no_grad()
    def from_qat(cls, qat_module: IntQuantLinear, pack_fn: Optional[Callable] = None) -> "IntQuantLinearInfra":
        prefix = qat_module.weight_quantizer.prefix if qat_module.weight_quantizer else ""
        infra_mod = cls(
            in_features=qat_module.in_features,
            out_features=qat_module.out_features,
            bias=qat_module.bias is not None,
            prefix=prefix, 
            config=qat_module.config
        ).to(qat_module.weight.device)

        infra_mod.weight.data.copy_(qat_module.weight.data)
        quantizer = qat_module.weight_quantizer
        if quantizer:
             scale, zp = quantizer.cal_qparams(quantizer.scale, quantizer.zero_point)
             # Reshape to [Out, GroupsPerRow]
             infra_mod.scales.data.copy_(scale.view(qat_module.out_features, -1))
             infra_mod.qzeros.data.copy_(zp.view(qat_module.out_features, -1))
        
        if qat_module.bias is not None:
            infra_mod.bias.data.copy_(qat_module.bias.data)
        return infra_mod

def convert_to_infra(module: nn.Module, pack_fn: Optional[Callable] = None, kernel_backend: Optional[Callable] = None) -> None:
    for name, child in module.named_children():
        if isinstance(child, IntQuantLinear):
            infra_layer = IntQuantLinearInfra.from_qat(child, pack_fn=pack_fn)
            infra_layer.kernel_backend = kernel_backend
            setattr(module, name, infra_layer)
        else:
            convert_to_infra(child, pack_fn=pack_fn, kernel_backend=kernel_backend)
