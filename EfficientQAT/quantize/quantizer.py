import torch
import torch.nn as nn
import random
import pdb

CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

# 感觉lsq+实现是这样的吗？
# def clamp_ste(x: torch.Tensor, min, max):
#     clamped = x.clamp(min, max)
#     gradient_mask = (x >= min) & (x <= max)  # 仅保留在范围内的梯度
#     return (clamped - x).detach() + x * gradient_mask


def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


class ClampMAD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        """
        前向传播：执行截断操作。
        """
        ctx.save_for_backward(x, min_val, max_val)
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：应用 MAD 规则调整梯度。
        """
        x, min_val, max_val = ctx.saved_tensors
        # 初始化梯度系数 alpha 为 1
        alpha = torch.ones_like(x)
        
        # 对于 x > max_val，alpha = max_val / x
        alpha = torch.where(x.abs() > max_val, max_val / x.abs(), alpha)
        
        # 对于 min_val <= x <= max_val，alpha 保持为 1
        grad_input = grad_output * alpha
        return grad_input, None, None

def clamp_mad(x: torch.Tensor, min_val, max_val):
    """
    使用 ClampMAD 进行截断操作。
    """
    return ClampMAD.apply(x, torch.tensor(min_val, device=x.device), torch.tensor(max_val, device=x.device))


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        args=None,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        self.clamp_method = args.clamp_method

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True)
                xmax =  x.amax([-1], keepdim=True)
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                if self.clamp_method == "STE":
                    scale = scale.clamp(min=1e-4, max=1e4)
                elif self.clamp_method == "MAD":
                    scale = clamp_mad(scale, 1e-4, 1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())
            

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self, x):
        
        if self.clamp_method == "STE":
            scale = clamp_ste(self.scale,1e-4, 1e4)
            round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        elif self.clamp_method == "MAD":
            scale = clamp_mad(self.scale, 1e-4, 1e4)
            round_zero_point = clamp_mad(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant

        

class GradualUniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        args=None,
        quantization_position_ratio=1.0,  # 新增量化比例参数，默认全量化
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        self.clamp_method = args.get("clamp_method", "STE")
        self.quantization_position_ratio = quantization_position_ratio  # 量化比例
        self.interpolate = 1.0 if args.get("interpolate", False) else 0  # 插值比例 0 代表没有前权重 1代表全是前权重

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1, self.group_size)
                xmin = x.amin([-1], keepdim=True)
                xmax = x.amax([-1], keepdim=True)
                range = xmax - xmin
                scale = range / (2**self.n_bits - 1)
                if self.clamp_method == "STE":
                    scale = scale.clamp(min=1e-4, max=1e4)
                elif self.clamp_method == "MAD":
                    scale = clamp_mad(scale, 1e-4, 1e4)
                zero_point = -(xmin / scale).clamp(min=-1e4, max=1e4)
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)

    def update_position_ratio(self, new_position_ratio):
        """Update the quantization position_ratio dynamically."""
        if 0.0 <= new_position_ratio <= 1.0:
            self.quantization_position_ratio = new_position_ratio
        else:
            raise ValueError("quantization_position_ratio should be between 0 and 1.")

    def update_interpolate_ratio(self, new_interpolate):
        """Update the interpolate ratio dynamically."""
        if 0.0 <= new_interpolate <= 1.0:
            self.interpolate = new_interpolate
        else:
            raise ValueError("interpolate should be between 0 and 1.")

    def fake_quant(self, x):
        if self.clamp_method == "STE":
            scale = clamp_ste(self.scale, 1e-4, 1e4)
            round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        elif self.clamp_method == "MAD":
            scale = clamp_mad(self.scale, 1e-4, 1e4)
            round_zero_point = clamp_mad(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)

        # 计算需要量化的组数
        total_groups = x.shape[0]
        quantized_groups = int(total_groups * self.quantization_position_ratio)
        if quantized_groups == 0: quantized_groups = 1  # 防止没有

        # 直接构造量化后的新张量
        x_quantized = x.clone()
        x_int = round_ste(x[:quantized_groups] / scale[:quantized_groups])
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point[:quantized_groups])
        x_int = x_int.clamp(self.qmin, self.qmax)

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point[:quantized_groups])
        x_dequant = x_dequant.mul(scale[:quantized_groups])

        # 返回量化后的新张量
        x_quantized[:quantized_groups] = x_dequant
        if self.interpolate>1e-6:
            x_quantized = (1-self.interpolate)*x_quantized+x*self.interpolate

        return x_quantized.reshape(dim1, dim2)

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        return self.fake_quant(x)