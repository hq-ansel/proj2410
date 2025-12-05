from typing import Dict,Callable,Iterable,List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantizer import(
    UniformAffineQuantizer,
    GradualQuantizer,
    QuantConfig
)

# 因为涉及到的weight quantizer有
def build_weight_quantizer(
    prefix: str,
    weight: torch.Tensor,
    config: QuantConfig
):
    """
    根据配置构建并返回相应的权重量化器实例
    
    Args:
        prefix (str): 量化器前缀标识符
        weight (torch.Tensor): 待量化的权重张量
        config (QuantConfig): 量化配置对象，包含量化类型等参数
    
    Returns:
        BaseQuantizer: 具体量化器实例，可能是UniformAffineQuantizer或GradualQuantizer
    
    Raises:
        ValueError: 当config.quant_type不是支持的量化类型时
    """
    if config.quant_type == "uniform_affine":
        return UniformAffineQuantizer(prefix,weight,config)
    elif config.quant_type == "gradual":
        return GradualQuantizer(prefix,weight,config)
    else:
        # fallback to uniform_affine 如果是无设置初始化
        # prefix 应该要能够通过上下文推测出来
        # weight和config应该是默认能够设置的
        return UniformAffineQuantizer(prefix,weight,config)
    pass

class IntQuantLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prefix: str = "",
        config: QuantConfig | None = None,
    ):
        # 1. 正常初始化 nn.Linear 的参数（weight, bias 等）
        super().__init__(in_features, out_features, bias=bias)

        # 2. 记录自己的额外信息（不会影响 state_dict 的 key 结构）
        self.use_weight_quant = False
        self.config = config

        # 3. 构建量化器（注意：这里用的是 self.weight）
        self.weight_quantizer = None
        if config is not None:
            self.weight_quantizer = build_weight_quantizer(
                prefix=prefix,
                weight=self.weight,
                config=config,
            )

    def forward(self, input):
        if self.use_weight_quant and self.weight_quantizer is not None:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)
    @classmethod
    def from_float(
        cls,
        prefix: str,
        org_module: nn.Linear,
        config: QuantConfig,
    ) -> "IntQuantLinear":
        # 1. 用 org_module 的形状信息正常创建一层新的 IntQuantLinear
        qlinear = cls(
            in_features=org_module.in_features,
            out_features=org_module.out_features,
            bias=org_module.bias is not None,
            prefix=prefix,
            config=config,
        ).to(org_module.weight.device, dtype=org_module.weight.dtype)
        qlinear.train(org_module.training)


        # 2. 拷贝权重（注意是 copy_，不是复用同一个 Parameter）
        qlinear.weight.data.copy_(org_module.weight.data)
        if org_module.bias is not None and qlinear.bias is not None:
            qlinear.bias.data.copy_(org_module.bias.data)

        return qlinear
    
def convert_linear(module: nn.Module, prefix: str, config: QuantConfig):
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if  isinstance(child, nn.Linear) and not isinstance(child, IntQuantLinear):
            setattr(module, name, IntQuantLinear.from_float(child_prefix, child, config))
        else:
            convert_linear(child, child_prefix, config)


def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            m.requires_grad = requires_grad
    return params

def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
            params.append(m)
    return params

def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            m.requires_grad = requires_grad
    return params

def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('scale') > -1 or n.find('zero_point') > -1:
            params.append(m)
    return params 

def set_quant_state(model, weight_quant: bool = False):
    for m in model.modules():
        if isinstance(m, IntQuantLinear):
            m.use_weight_quant = weight_quant
            
@torch.no_grad()   
def quant_inplace(model):
    for _, m in model.named_modules():
        if isinstance(m, IntQuantLinear):
            m.weight.data = m.weight_quantizer(m.weight.data)

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)