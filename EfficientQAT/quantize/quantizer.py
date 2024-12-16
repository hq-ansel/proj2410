from unittest import skip
import torch
import torch.nn as nn
import random
import pdb

CLIPMIN = 1e-4

HighPassThreshold = 1e-1


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

class HighPassRoundSTE(torch.autograd.Function):
    """
    result = round(x)
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        res = x.round() - x
        # res \in (-0.5, 0.5)
        result = torch.where(res.abs() > HighPassThreshold, x.round(), x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output
        return grad_input, None

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
        self.clamp_method = args.get("clamp_method", "STE")

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


class UniformAffineQuantizerV2(nn.Module):
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
        self.clamp_method = args.get("clamp_method", "STE")

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
            x_int = x_int.add(round_zero_point*0.9)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point*0.9)
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

        if self.clamp_method == "STE":
            self.clamp_method =clamp_ste
        elif self.clamp_method == "MAD":
            self.clamp_method = clamp_mad

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

    def quant_int(self, x, scale, zero_point):
        """
        隐式要求x.shape[-1] % self.group_size == 0
        """
        x_int = round_ste(x / scale)
        if zero_point is not None:
            x_int = x_int.add(zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        return x_int
    
    def dequant_int(self, x_int, scale, zero_point):
        x_dequant = x_int
        if zero_point is not None:
            x_dequant = x_dequant.sub(zero_point)
        x_dequant = x_dequant.mul(scale)
        return x_dequant
    
    def fake_quant(self, x):

        scale = self.clamp_method(self.scale, 1e-4, 1e4)
        round_zero_point = self.clamp_method( round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)

        # 计算需要量化的组数
        total_groups = x.shape[0]
        quantized_groups = int(total_groups * self.quantization_position_ratio)
        if quantized_groups == 0: quantized_groups = 1  # 防止没有

        # 直接构造量化后的新张量
        x_quantized = x.clone()
        # x_int = round_ste(x[:quantized_groups] / scale[:quantized_groups])
        # if round_zero_point is not None:
        #     x_int = x_int.add(round_zero_point[:quantized_groups])
        # x_int = x_int.clamp(self.qmin, self.qmax)
        x_int = self.quant_int(x[:quantized_groups], scale[:quantized_groups], round_zero_point[:quantized_groups])

        x_dequant = x_int
        # if round_zero_point is not None:
        #     x_dequant = x_dequant.sub(round_zero_point[:quantized_groups])
        # x_dequant = x_dequant.mul(scale[:quantized_groups])
        x_dequant = self.dequant_int(x_int, scale[:quantized_groups], round_zero_point[:quantized_groups])

        # 返回量化后的新张量
        x_quantized[:quantized_groups] = x_dequant
        if self.interpolate>1e-6:
            x_quantized = (1-self.interpolate)*x_quantized+x*self.interpolate

        return x_quantized.reshape(dim1, dim2)

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        return self.fake_quant(x)
    
    def get_inferred_params(self,x):
        scale = clamp_ste(self.scale, 1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)

        # 计算需要量化的组数
        total_groups = x.shape[0]
        quantized_groups = int(total_groups * self.quantization_position_ratio)
        if quantized_groups == 0: quantized_groups = 1  # 防止没有

        # 直接构造量化后的新张量
        x_int = round_ste(x[:quantized_groups] / scale[:quantized_groups])
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point[:quantized_groups])
        x_int = x_int.clamp(self.qmin, self.qmax)

        return x_int.reshape(dim1, dim2), scale, round_zero_point


class GradualUniformAffineQuantizerV2(nn.Module):
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

        self.is_tracking = args.get("iterative_freezing", False)
        self.weight_freeze_tracker = TrackOscillation(
            momentum=args.get("freeze_momentum",0.004),
            freeze_threshold=args.get("freeze_threshold",0.0),
            use_ema_x_int=True
            )

        if self.clamp_method == "STE":
            self.clamp_method =clamp_ste
        elif self.clamp_method == "MAD":
            self.clamp_method = clamp_mad

        if args.get("round_method", "ste"):
            self.round_method = round_ste
        elif args.get("round_method", "highpass"):
            self.round_method = HighPassRoundSTE.apply  # 新增高通滤波器
        print("round_method:", args.get("round_method", "ste"))
        # self.round_method = lambda x: round_ste(x)

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1, self.group_size)
                xmin = x.amin([-1], keepdim=True)
                xmax = x.amax([-1], keepdim=True)
                range = xmax - xmin
                scale = range / (2**self.n_bits - 1)
                scale = self.clamp_method(scale, 1e-4, 1e4)
                zero_point = -(xmin / scale).clamp(min=-1e4, max=1e4)
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())

    def extra_repr(self):
        return (
            f"n_bits={self.n_bits}, "
            f"group_size={self.group_size}, "
            f"quantization_position_ratio={self.quantization_position_ratio}, "
            f"interpolate={self.interpolate}, "
            f"weight_freeze_tracker={self.weight_freeze_tracker}"
        )

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

    def quant_int(self, x, scale, zero_point):
        """
        隐式要求x.shape[-1] % self.group_size == 0
        """
        # x_int = round_ste(x / scale)
        x_int = self.round_method(x / scale)
        if zero_point is not None:
            x_int = x_int.add(zero_point)
        # 这个地方有三种写法
        # 1. x_int = x_int.clamp(self.qmin, self.qmax) 代表着范围意外的梯度截断
        # 2. x_int = clamp_ste(x_int, self.qmin, self.qmax) 代表着基本ste
        # 3. x_int = clamp_mad(x_int, self.qmin, self.qmax) 代表着mad
        x_int = x_int.clamp(self.qmin, self.qmax)
        # x_int = clamp_ste(x_int, self.qmin, self.qmax) 
        # x_int = clamp_mad(x_int, self.qmin, self.qmax)
        return x_int
    
    def dequant_int(self, x_int, scale, zero_point):
        x_dequant = x_int
        if zero_point is not None:
            x_dequant = x_dequant.sub(zero_point)
        x_dequant = x_dequant.mul(scale)
        return x_dequant
    
    def fake_quant(self, x):

        scale = self.clamp_method(self.scale, 1e-4, 1e4)
        round_zero_point = self.clamp_method(self.round_method(self.zero_point), self.qmin, self.qmax)
        # round_zero_point = self.clamp_method(self.zero_point.round(), self.qmin, self.qmax)
        # round_zero_point = self.clamp_method(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)

        # 计算需要量化的组数
        total_groups = x.shape[0]
        quantized_groups = int(total_groups * self.quantization_position_ratio)
        if quantized_groups == 0: quantized_groups = 1  # 防止没有

        # 直接构造量化后的新张量
        x_quantized = x.clone()
        x_int = self.quant_int(x[:quantized_groups], scale[:quantized_groups], round_zero_point[:quantized_groups])
        x_dequant = x_int

        # freezing weights
        x_int = self.weight_freeze_tracker(x_int,skip_tracking=False)
        x_dequant = self.dequant_int(x_int, scale[:quantized_groups], round_zero_point[:quantized_groups])

        # 返回量化后的新张量
        x_quantized[:quantized_groups] = x_dequant
        if self.interpolate>1e-6:
            x_quantized = (1-self.interpolate)*x_quantized+x*self.interpolate

        return x_quantized.reshape(dim1, dim2)

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        return self.fake_quant(x)
    

    def get_inferred_params(self,x):
        scale = clamp_ste(self.scale, 1e-4, 1e4)
        round_zero_point = clamp_ste(self.zero_point.round(), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)

        # 计算需要量化的组数
        total_groups = x.shape[0]
        quantized_groups = int(total_groups * self.quantization_position_ratio)
        if quantized_groups == 0: quantized_groups = 1  # 防止没有

        # 直接构造量化后的新张量
        x_int = round_ste(x[:quantized_groups] / scale[:quantized_groups])
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point[:quantized_groups])
        x_int = x_int.clamp(self.qmin, self.qmax)

        return x_int.reshape(dim1, dim2), scale, round_zero_point


class TrackOscillation(torch.nn.Module):
    """
    mainly refer to https://github.com/nbasyl/OFQ/blob/7ed37d1dd33d39395edbf49fcbbc52f678ecf961/src/quantization/quantizer/lsq.py#L111
    and https://github.com/Qualcomm-AI-research/oscillations-qat/blob/9064d8540c1705242f08b864f06661247012ee4d/utils/oscillation_tracking_utils.py#L26
    This is a wrapper of the int_forward function of a quantizer.
    It tracks the oscillations in integer domain.
    """

    def __init__(self, momentum=0.01, freeze_threshold=0, use_ema_x_int=True):
        super(TrackOscillation, self).__init__()
        self.momentum = momentum

        self.prev_x_int = None
        self.prev_switch_dir = None

        # Statistics to log
        self.ema_oscillation = None
        self.oscillated_sum = None
        self.total_oscillation = None
        self.iters_since_reset = 0

        # Extra variables for weight freezing
        self.freeze_threshold = freeze_threshold  # This should be at least 2-3x the momentum value.
        self.use_ema_x_int = use_ema_x_int
        self.frozen = None
        self.frozen_x_int = None
        self.ema_x_int = None

    def __call__(self, x_int, skip_tracking=False, *args, **kwargs):
       
        # Apply weight freezing
        if self.frozen is not None:
            x_int = ~self.frozen * x_int + self.frozen * self.frozen_x_int

        if skip_tracking:
            return x_int

        with torch.no_grad():
            # Check if everything is correctly initialized, otherwise do so
            self.check_init(x_int)

            # detect difference in x_int  NB we round to avoid int inaccuracies
            delta_x_int = torch.round(self.prev_x_int - x_int).detach()  # should be {-1, 0, 1}
            switch_dir = torch.sign(delta_x_int)  # This is {-1, 0, 1} as sign(0) is mapped to 0
            # binary mask for switching
            switched = delta_x_int != 0

            oscillated = (self.prev_switch_dir * switch_dir) == -1
            self.ema_oscillation = (
                self.momentum * oscillated + (1 - self.momentum) * self.ema_oscillation
            )

            # Update prev_switch_dir for the switch variables
            self.prev_switch_dir[switched] = switch_dir[switched]
            self.prev_x_int = x_int
            self.oscillated_sum = oscillated.sum()
            self.total_oscillation += oscillated
            self.iters_since_reset += 1

            # Freeze some weights
            if self.freeze_threshold > 0:
                freeze_weights = self.ema_oscillation > self.freeze_threshold
                self.frozen[freeze_weights] = True  # Set them to frozen
                if self.use_ema_x_int:
                    self.frozen_x_int[freeze_weights] = torch.round(self.ema_x_int[freeze_weights])
                    # Update x_int EMA which can be used for freezing
                    self.ema_x_int = self.momentum * x_int + (1 - self.momentum) * self.ema_x_int
                else:
                    self.frozen_x_int[freeze_weights] = x_int[freeze_weights]

        return x_int

    def check_init(self, x_int):
        if self.prev_x_int is None:
            # Init prev switch dir to 0
            self.prev_switch_dir = torch.zeros_like(x_int)
            self.prev_x_int = x_int.detach()  # Not sure if needed, don't think so
            self.ema_oscillation = torch.zeros_like(x_int)
            self.oscillated_sum = 0
            self.total_oscillation = torch.zeros_like(x_int)
            # print("Init tracking", x_int.shape)
        else:
            assert (
                self.prev_x_int.shape == x_int.shape
            ), "Tracking shape does not match current tensor shape."

        # For weight freezing
        if self.frozen is None and self.freeze_threshold > 0:
            self.frozen = torch.zeros_like(x_int, dtype=torch.bool)
            self.frozen_x_int = torch.zeros_like(x_int)
            if self.use_ema_x_int:
                self.ema_x_int = x_int.detach().clone()