import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import (UniformAffineQuantizer,GradualUniformAffineQuantizer,GradualUniformAffineQuantizerV2,
                        UniformAffineQuantizerV2)
from . import quantizer,quantizerv2,quantizerv3




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
        # get (out_features, in_features)
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
        quantizer_version = args.get("quantizer_version","v1")
        if quantizer_version == "v1":
            quantizer_pkg = quantizer
        elif quantizer_version == "v2":
            quantizer_pkg = quantizerv2
        elif quantizer_version == "v3":
            quantizer_pkg = quantizerv3
        else:
            raise ValueError("Invalid quantizer version: {}".format(quantizer_version))
        # 确定 quantizer的具体版本
        if args.get("gradual_quant", False):
            self.weight_quantizer = quantizer_pkg.GradualUniformAffineQuantizer(wbits,
                                                            group_size,
                                                            weight=org_module.weight,
                                                            args=args)
        elif args.get("iterative_freezing", False):
            self.weight_quantizer = quantizer_pkg.GradualUniformAffineQuantizerV2(wbits,
                                                                     group_size,
                                                                     weight=org_module.weight,
                                                                     args=args)
        elif args.get("dsq", False):
            self.weight_quantizer = quantizer_pkg.DSQuantizer(wbits,
                                                               group_size,
                                                               weight=org_module.weight,
                                                               args=args)
        else:
            self.weight_quantizer = quantizer_pkg.UniformAffineQuantizer(wbits, group_size, weight=org_module.weight,args=args)
            # self.weight_quantizer = UniformAffineQuantizerV2(wbits, group_size, weight=org_module.weight,args=args)
        self.use_temporary_parameter = False
        self.clamp_input = args.get('clamp_input',False)
        self.post_init(args)

    def post_init(self, args):
        quantizer_version = args.get("quantizer_version","v1")
        if quantizer_version == "v3":
            with torch.no_grad():
                updated_weight = self.weight_quantizer.post_init(self.weight)
                self.weight.data.copy_(updated_weight)
    
    
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # out = torch.matmul(input, weight.T)

        return out

    def get_dampen_loss(self):

        return torch.norm(
            self.weight_quantizer.fake_quant(self.weight).detach() - self.weight,
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