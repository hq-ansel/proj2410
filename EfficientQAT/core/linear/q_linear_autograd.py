import torch
from torch.amp import custom_bwd, custom_fwd

from .q_linear_triton_kernels import quant_matmul


class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq):
        output = quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq, ctx.pack_bits = bits, maxq, pack_bits
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq, pack_bits = ctx.bits, ctx.maxq, ctx.pack_bits
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul(
                grad_output, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=True
            )
        return grad_input, None, None, None, None, None, None


__all__ = ["QuantLinearFunction"]
