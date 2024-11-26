import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import UniformAffineQuantizer,GradualUniformAffineQuantizer,GradualUniformAffineQuantizerV2





class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64,
        args=None,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        # initialize quantizer
        # self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight,args=args)
        if args.get("gradual_quant", False):
            self.weight_quantizer = GradualUniformAffineQuantizer(wbits,
                                                            group_size,
                                                            weight=org_module.weight,
                                                            args=args)
        elif args.get("iterative_freezing", False):
            self.weight_quantizer = GradualUniformAffineQuantizerV2(wbits,
                                                                     group_size,
                                                                     weight=org_module.weight,
                                                                     args=args)
        else:
            self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight,args=args)
        self.use_temporary_parameter = False
        self.clamp_input = args.get('clamp_input',False)

    
    
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.clamp_input:
            input = torch.clamp(input,-128,127)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out
    def get_dampen_loss(self):
        def clamp_ste(x: torch.Tensor, _min, _max):
            return (x.clamp(_min,_max) - x).detach() + x
        zero_point = self.weight_quantizer.zero_point
        scale = self.weight_quantizer.scale
        a_min = zero_point*scale
        a_max = (2**self.weight_quantizer.n_bits-1)*scale+a_min

        return torch.norm(
            self.weight_quantizer.fake_quant(self.weight).reshape(-1,a_min.shape[-1]) -
                clamp_ste(self.weight,a_min, a_max),
            p=2
        )

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant

    def get_quant_weight_bias(self):
        weight = self.weight_quantizer(self.weight)
        bias = self.bias
        return weight, bias

    def update_position_ratio(self, ratio: float):
        """
        Update the quantization ratio of the weight.
        """
        self.weight_quantizer.update_position_ratio(ratio)
    def update_interpolate_ratio(self, ratio: float):
        """
        Update the interpolation ratio of the raw weight.
        """
        self.weight_quantizer.update_interpolate_ratio(ratio)

    def get_inferred_params(self):
        int_weight,scale,zero_point = self.weight_quantizer.get_inferred_params(self.weight)
        return int_weight,scale,zero_point