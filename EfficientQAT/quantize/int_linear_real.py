from logging import getLogger
import math
import gc  
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model

from .triton_utils.kernels import dequant_dim0, dequant_dim1
from .utils import get_named_linears,set_op_by_name

from gptqmodel.nn_modules.qlinear.qlinear_tritonv2 import TritonV2QuantLinear

logger = getLogger(__name__)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        **kwargs
    ):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        if infeatures % 32 != 0 or outfeatures % 32 != 0:
            raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.in_features = infeatures
        self.outfeatures = outfeatures
        self.out_features = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer(
            'qweight',
            torch.zeros((math.ceil(infeatures / (32 // self.bits)), outfeatures), dtype=torch.int32)
        )
        self.register_parameter(
            'scales',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(infeatures / self.group_size), math.ceil(outfeatures / (32 // self.bits))), dtype=torch.int32)
        )
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )   # not used, just for consistent with GPTQ models
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.use_fake = False
        self.clamp_input = kwargs.get("clamp_input", False)

    def post_init(self):
        pass


    def use_fake_quantization(self, del_quant=False,transpose=False):
        # use fake quantization for faster training but consume more memory
        weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        dim0, dim1 = weight.shape
        zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
        weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1).contiguous()
        self.register_buffer(
            'weight',
            weight
        )
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.qzeros
            del self.g_idx
        
    def pack(self, linear, scales, zeros, g_idx=None):
        """
        Args:
            linear: nn.Linear 
            scales: scales tensor of shape (infeatures//group_size, outfeatures)
            zeros: zeros tensor of shape (infeatures//group_size, outfeatures)
        """
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
    
        g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32)

        scale_zeros = zeros * scales
        self.scales = nn.Parameter(scales.half())
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx] + scale_zeros[g_idx[idx]]) / self.scales[g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        # qweight (infeatures//(32//bits), outfeatures)
        # intweight (infeatures, outfeatures)
        qweight = np.zeros((math.ceil(intweight.shape[0]/(32//self.bits)), intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros = zeros.numpy().astype(np.uint32)
        self.zeros_dim0, self.zeros_dim1 = zeros.shape
        # qzeros (infeatures//group_size, outfeatures//(32//bits))
        qzeros = np.zeros((zeros.shape[0], math.ceil(zeros.shape[1] / (32 // self.bits))), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), zeros.shape[1])):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
            dim0, dim1 = weight.shape
            # dim2 = (dim1*dim0)//self.group_size
            zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
            weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        # out = torch.matmul(x, weight)
        # torch.cuda.synchronize()
        # if self.clamp_input:
        #     x = torch.clamp(x, -128, 127)
        out = torch.matmul(x, weight.to(x.dtype))
        # torch.cuda.synchronize()

        # out = out + self.bias.to(x.dtype) if self.bias is not None else out
        if self.bias is not None:
            out = out + self.bias.to(out.device, dtype=out.dtype)
        return out
    
    @classmethod
    @staticmethod
    def from_TritonV2QuantLinear(linear: TritonV2QuantLinear):
        q_linear = QuantLinear(linear.bits, 
                                linear.group_size, 
                                linear.infeatures, 
                                linear.outfeatures, 
                                linear.bias is not None,
        )
        q_linear.qweight = linear.qweight
        q_linear.qzeros = linear.qzeros
        q_linear.scales = torch.nn.Parameter(linear.scales).to(linear.scales.dtype)
        q_linear.g_idx = linear.g_idx
        if linear.bias is not None:
            q_linear.bias = torch.nn.Parameter(linear.bias.clone())
        del linear
        return q_linear


# V2 子类，继承 QuantLinear
class QuantLinearV2(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton_v2"  # 可以改变 QUANT_TYPE 或其他常量

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        **kwargs
    ):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        if infeatures % 32 != 0 or outfeatures % 32 != 0:
            raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.in_features = infeatures
        self.outfeatures = outfeatures
        self.out_features = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer(
            'qweight',
            torch.zeros((math.ceil(infeatures / (32 // self.bits)), outfeatures), dtype=torch.int32)
        )
        self.register_parameter(
            'scales',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        )
        self.register_buffer(
            'qzeros',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        )
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )   # not used, just for consistent with GPTQ models
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.use_fake = False
        self.clamp_input = kwargs.get("clamp_input", False)

    def post_init(self):
        pass


    def use_fake_quantization(self, del_quant=False,transpose=False):
        # use fake quantization for faster training but consume more memory
        weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        dim0, dim1 = weight.shape
        # zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
        zeros = self.qzeros
        # weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        weight = (weight.view(-1, self.group_size, dim1) * self.scales.view(-1, 1, dim1) - zeros.view(-1, 1, dim1)).reshape(dim0, dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1).contiguous()
        self.register_buffer(
            'weight',
            weight
        )
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.qzeros
            del self.g_idx
        
    def pack(self, linear, scales, zeros, g_idx=None):
        """
        Args:
            linear: nn.Linear (out,in)
            scales: scales tensor of shape (infeatures//group_size, outfeatures)
            zeros: zeros tensor of shape (infeatures//group_size, outfeatures)
        """
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
    
        g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32)

        self.scales = nn.Parameter(scales.half())
        self.qzeros = nn.Parameter(zeros.half())

        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx]+zeros[g_idx[idx]]) / self.scales[g_idx[idx]]
                ).clamp(0,self.maxq).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        # intweight now is (infeatures, outfeatures)

        i = 0
        row = 0
        # qweight (infeatures//(32//bits), outfeatures)
        # intweight (infeatures, outfeatures)
        qweight = np.zeros((math.ceil(intweight.shape[0]/(32//self.bits)), intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        self.zeros_dim0, self.zeros_dim1 = zeros.shape
        # qzeros (infeatures//group_size, outfeatures//(32//bits))

    def forward(self, x):
        if self.use_fake:
            # 默认linear是(out,in)
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
            dim0, dim1 = weight.shape
            # dim2 = (dim1*dim0)//self.group_size
            zeros = self.qzeros
            # weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
            weight = (weight.view(-1, self.group_size, dim1) * self.scales.view(-1, 1, dim1) - zeros.view(-1, 1, dim1)).reshape(dim0, dim1)

        out = torch.matmul(x, weight.to(x.dtype))

        if self.bias is not None:
            out = out + self.bias.to(out.device, dtype=out.dtype)
        return out
    
    @classmethod
    @staticmethod
    def from_TritonV2QuantLinear(linear: TritonV2QuantLinear):
        q_linear = QuantLinear(linear.bits, 
                                linear.group_size, 
                                linear.infeatures, 
                                linear.outfeatures, 
                                linear.bias is not None,
        )
        q_linear.qweight = linear.qweight
        q_linear.qzeros = linear.qzeros
        q_linear.scales = torch.nn.Parameter(linear.scales).to(linear.scales.dtype)
        q_linear.g_idx = linear.g_idx
        if linear.bias is not None:
            q_linear.bias = torch.nn.Parameter(linear.bias.clone())
        del linear
        return q_linear





def load_quantized_model(model_path, wbits, group_size):
    print(f"Loading quantized model from {model_path}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            if "v2" in model_path:
                q_linear = QuantLinearV2(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            else:
                q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    # print("Loading pre-computed quantized weights...",model)
    model.tie_weights()
    # kwargs = {"max_memory": "16GB"} 
    device_map = infer_auto_device_map(model,
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "Qwen2DecoderLayer",
            "MPTBlock",
            "DecoderLayer",
        ],                               
        # **kwargs
        )
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,
        checkpoint=model_path,
        device_map=device_map,
        offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer

__all__ = ["QuantLinear","load_quantized_model", "QuantLinearV2"]
