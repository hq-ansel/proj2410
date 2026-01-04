# ops.py
"""量化操作相关的自定义自动微分函数和工具函数"""
import torch
from torch import Tensor

# 高通滤波阈值
HighPassThreshold = 1e-1

def round_ste(x: Tensor) -> Tensor:
    """直通估计器（Straight-Through Estimator）舍入

    Args:
        x: 输入张量

    Returns:
        舍入后的张量（前向传播）和原始输入（反向传播）
    """
    return (x.round() - x).detach() + x

class HighPassRoundSTE(torch.autograd.Function):
    """高通滤波舍入 STE - 仅对大误差进行舍入"""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """前向传播：对误差大于阈值的部分进行舍入

        Args:
            x: 输入张量

        Returns:
            处理后的张量
        """
        ctx.save_for_backward(x)
        res = x.round() - x
        return torch.where(res.abs() > HighPassThreshold, x.round(), x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """反向传播：直通估计器

        Args:
            grad_output: 梯度输出

        Returns:
            梯度输入
        """
        (x,) = ctx.saved_tensors
        return grad_output


def clamp_ste(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """带直通估计器的截断

    Args:
        x: 输入张量
        min_val: 最小值
        max_val: 最大值

    Returns:
        截断后的张量（前向传播）和原始输入（反向传播）
    """
    return (x.clamp(min_val, max_val) - x).detach() + x


class ClampMAD(torch.autograd.Function):
    """带绝对值偏差（MAD）的自适应截断"""

    @staticmethod
    def forward(ctx, x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
        """前向传播：截断输入

        Args:
            x: 输入张量
            min_val: 最小值张量
            max_val: 最大值张量

        Returns:
            截断后的张量
        """
        ctx.save_for_backward(x, min_val, max_val)
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """反向传播：基于 MAD 自适应调整梯度

        Args:
            grad_output: 梯度输出

        Returns:
            输入梯度、最小值梯度（None）、最大值梯度（None）
        """
        x, min_val, max_val = ctx.saved_tensors
        alpha = torch.ones_like(x)
        # 仅对超过最大值的元素调整梯度
        alpha = torch.where(x.abs() > max_val, max_val / x.abs(), alpha)
        grad_input = grad_output * alpha
        return grad_input, None, None


def clamp_mad(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """使用 MAD 自适应截断

    Args:
        x: 输入张量
        min_val: 最小值
        max_val: 最大值

    Returns:
        MAD 截断后的张量
    """
    return ClampMAD.apply(
        x,
        torch.tensor(min_val, device=x.device),
        torch.tensor(max_val, device=x.device),
    )
