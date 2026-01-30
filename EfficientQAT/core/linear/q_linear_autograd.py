import torch
from torch.amp import custom_bwd, custom_fwd

from .q_linear_triton_kernels import quant_matmul


class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, sym: bool = False):
        output = quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, sym=sym)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq, ctx.pack_bits, ctx.sym = bits, maxq, pack_bits, sym
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq, pack_bits, sym = ctx.bits, ctx.maxq, ctx.pack_bits, ctx.sym
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul(
                grad_output, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=True, sym=sym
            )
        return grad_input, None, None, None, None, None, None, None, None


__all__ = ["QuantLinearFunction"]
