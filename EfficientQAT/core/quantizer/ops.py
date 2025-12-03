# ops.py
import torch
from torch import Tensor

HighPassThreshold = 1e-1

def round_ste(x: Tensor) -> Tensor:
    return (x.round() - x).detach() + x

class HighPassRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        res = x.round() - x
        return torch.where(res.abs() > HighPassThreshold, x.round(), x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        return grad_output
        

def clamp_ste(x: Tensor, min_val: float, max_val: float) -> Tensor:
    return (x.clamp(min_val, max_val) - x).detach() + x


class ClampMAD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
        ctx.save_for_backward(x, min_val, max_val)
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, min_val, max_val = ctx.saved_tensors
        alpha = torch.ones_like(x)
        alpha = torch.where(x.abs() > max_val, max_val / x.abs(), alpha)
        grad_input = grad_output * alpha
        return grad_input, None, None


def clamp_mad(x: Tensor, min_val: float, max_val: float) -> Tensor:
    return ClampMAD.apply(
        x,
        torch.tensor(min_val, device=x.device),
        torch.tensor(max_val, device=x.device),
    )
