
import torch
import torch.nn as nn
from torch.nn import (
    Linear
)
import torch.nn.functional as F
import torch.nn.init as init



def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


class BaseQuantizer(nn.Module):
    def __init__(self, 
        n_bits: int = 8,
        group_size=None,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size 

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)

    @staticmethod
    def init_with_weight(weight, n_bits, group_size):
        if weight.dtype == torch.float32 or weight.dtype == torch.float16:
            scale_dtype = torch.float16
        else:
            scale_dtype = torch.bfloat16
        with torch.no_grad():
            x = weight.reshape(-1,group_size)
            xmin = x.amin([-1], keepdim=True)
            xmax =  x.amax([-1], keepdim=True)
            x_range = xmax - xmin
            scale = x_range / (2**n_bits-1)
            scale = scale.clamp(min=1e-4, max=1e4)
            zero_point = -xmin/scale
            return scale.to(scale_dtype), zero_point.round().to(scale_dtype)
        
    def cal_qparams(self,scale,zero_point,clamp_method="STE"):
        if clamp_method == "STE":
            scale_dtype = scale.dtype
            scale = clamp_ste(scale,1e-4, 1e4).to(scale_dtype)
            round_zero_point = clamp_ste(round_ste(zero_point), self.qmin, self.qmax)
        return scale, round_zero_point

    def _quantize(self, x, scale, round_zero_point):
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        return x_int
    
    def _dequantize(self, x_int, scale, round_zero_point):
        if round_zero_point is not None:
            x_int = x_int.sub(round_zero_point)
        x_float = x_int * scale
        return x_float
    
    def fake_quant(self, x):
        scale, round_zero_point = self.cal_qparams(self.scale,
                                                   self.zero_point,)
        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = self._quantize(x, scale, round_zero_point)
        x_dequant = self._dequantize(x_int, scale, round_zero_point)
        return x_dequant.reshape(ori_shape)
    
    
        

class UniformAffineQuantizer(BaseQuantizer):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        args=None,
    ):
        super().__init__(
            n_bits=n_bits,
            group_size=group_size,
        )
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        scale, zero_points = BaseQuantizer.init_with_weight(
            weight, n_bits, group_size)
        self.scale = nn.Parameter(scale)
        self.zero_point = nn.Parameter(zero_points)

        self.enable = True

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant



class QuantLinearFake(nn.Module):
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
        self.use_weight_quant = False

        self.weight_quantizer = UniformAffineQuantizer(wbits, group_size,
                                                        weight=org_module.weight,args=args)
    
    def forward(self, x: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.fwd_func(x, weight, bias, **self.fwd_kwargs)
        return out,weight,x

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant

    def get_quant_weight_bias(self):
        weight = self.weight_quantizer(self.weight)
        bias = self.bias
        return weight, bias

    def get_inferred_params(self):
        int_weight,scale,zero_point = self.weight_quantizer.get_inferred_params(self.weight)
        return int_weight,scale,zero_point
    

import numpy as np
import math
import time
import builtins
from typing import Dict

import triton
import triton.language as tl


class CustomizedTritonAutoTuner(triton.KernelInterface):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        prune_configs_by: Dict = None,
        nearest_power_of_two: bool = False
    ):
        if not configs:
            self.configs = [triton.Config({}, num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.nearest_power_of_two = nearest_power_of_two
        self.cache = {}
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()

            self.hook = _hook
        self.arg_names = arg_names
        # prune configs
        if prune_configs_by:
            perf_model, top_k = prune_configs_by['perf_model'], prune_configs_by['top_k']
            if 'early_config_prune' in prune_configs_by:
                early_config_prune = prune_configs_by['early_config_prune']
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.fn = fn

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.kwargs)

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(self.nargs)
            self.hook(args)
            self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, **current)

        try:
            # In testings using only 40 reps seems to be close enough and it appears to be what PyTorch uses
            # PyTorch also sets fast_flush to True, but I didn't see any speedup so I'll leave the default
            return triton.testing.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8), rep=40)
        except triton.OutOfResources:
        # except triton.OutOfResources:
            return (float('inf'), float('inf'), float('inf'))

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            key = tuple(args[i] for i in self.key_idx)

            # This reduces the amount of autotuning by rounding the keys to the nearest power of two
            # In my testing this gives decent results, and greatly reduces the amount of tuning required
            if self.nearest_power_of_two:
                key = tuple([2 ** int(math.log2(x) + 0.5) for x in key])

            if key not in self.cache:
                # prune configs
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.hook(args)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if config.pre_hook is not None:
            config.pre_hook(self.nargs)
        return self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, **kwargs, **config.kwargs)

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(**self.nargs, **kwargs, **config.kwargs, num_stages=config.num_stages,
                                            num_warps=config.num_warps) for config in pruned_configs}
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        for config in self.prune_configs(kwargs):
            self.fn.warmup(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                **kwargs,
                **config.kwargs,
            )
        self.nargs = None


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, nearest_power_of_two=False):
    def decorator(fn):
        return CustomizedTritonAutoTuner(
            fn, fn.arg_names, configs, key, reset_to_zero, prune_configs_by, nearest_power_of_two
        )

    return decorator

def hadamard248_kernel_config_pruner(configs, nargs):
    """
    The main purpose of this function is to shrink BLOCK_SIZE_* when the corresponding dimension is smaller.
    """
    m = max(2 ** int(math.ceil(math.log2(nargs['M']))), 16)
    n = max(2 ** int(math.ceil(math.log2(nargs['N']))), 16)

    used = set()
    for config in configs:
        block_size_m = min(m, config.kwargs['BLOCK_SIZE_M'])
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])

        if (block_size_m, block_size_n , config.num_stages, config.num_warps) in used:
            continue

        used.add((block_size_m, block_size_n, config.num_stages, config.num_warps))
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps
        )


@autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=2,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=4,
            num_warps=4
        ),
    ],
    key=['M', 'N'],
    nearest_power_of_two=True,
    prune_configs_by={
        'early_config_prune': hadamard248_kernel_config_pruner,
        'perf_model': None,
        'top_k': None,
    },
)
@triton.jit
def dequant_kernel_dim0(
    b_ptr, c_ptr,
    M, N,
    bits, maxq,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    dequant the quantized tensor to fp tensor
    B is of shape (M/(32//bits), N) int32
    C is of shape (M, N) float16
    """

    bits_per_feature = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    tl.device_assert(BLOCK_SIZE_M % bits_per_feature == 0)
    tl.device_assert(BLOCK_SIZE_N % bits_per_feature == 0)

    tl.device_assert(M % BLOCK_SIZE_M == 0)
    tl.device_assert(N % BLOCK_SIZE_N == 0)

    b_ptrs = b_ptr + ((offs_am[:, None] // bits_per_feature) * stride_bk + offs_bn[None, :] * stride_bn)

    shifter = (offs_am[:, None] % bits_per_feature) * bits



    b = tl.load(b_ptrs)
    b = (b >> shifter) & maxq
  
    c = b

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N':128,
            },
            num_stages=8,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N':64,
            },
            num_stages=8,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N':32,
            },
            num_stages=8,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N':2,
            },
            num_stages=8,
            num_warps=8
        ),
    ],
    key=['M', 'N'],
    nearest_power_of_two=True,
    prune_configs_by={
        'early_config_prune': hadamard248_kernel_config_pruner,
        'perf_model': None,
        'top_k': None,
    },
)
@triton.jit
def dequant_kernel_dim1(
    b_ptr, c_ptr,
    M, N,
    bits, maxq,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    dequant the quantized tensor to fp tensor
    B is of shape (M, N/(32//bits)) int32
    C is of shape (M, N) float16
    """

    bits_per_feature = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)


    tl.device_assert(BLOCK_SIZE_M % bits_per_feature == 0)
    tl.device_assert(BLOCK_SIZE_N % bits_per_feature == 0)

    tl.device_assert(M % BLOCK_SIZE_M == 0)
    tl.device_assert(N % BLOCK_SIZE_N == 0)

    # b_ptrs = b_ptr + ((offs_am[:, None] // bits_per_feature) * stride_bk + offs_bn[None, :] * stride_bn)
    b_ptrs = b_ptr + (offs_am[:, None] * stride_bk + (offs_bn[None, :] // bits_per_feature) * stride_bn)

    # shifter = (offs_am[:, None] % bits_per_feature) * bits
    shifter = (offs_bn[None, :] % bits_per_feature) * bits


    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

    b = tl.load(b_ptrs,mask=c_mask)
    b = (b >> shifter) & maxq
  
    c = b

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)

def dequant_dim0(qweight, bits, maxq, infeatures, outfeatures, dtype=torch.float16):
    with torch.cuda.device(qweight.device):
        output = torch.empty((infeatures, outfeatures), device=qweight.device, dtype=dtype)
        grid = lambda META: (
            triton.cdiv(output.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output.shape[1], META['BLOCK_SIZE_N']),
        )
        dequant_kernel_dim0[grid](
            qweight, output,
            output.shape[0], output.shape[1],
            bits, maxq,
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

def dequant_dim1(qweight, bits, maxq, infeatures, outfeatures, dtype=torch.float16):
    with torch.cuda.device(qweight.device):
        output = torch.empty((infeatures, outfeatures), device=qweight.device, dtype=dtype)
        grid = lambda META: (
            triton.cdiv(output.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output.shape[1], META['BLOCK_SIZE_N']),
        )
        dequant_kernel_dim1[grid](
            qweight, output,
            output.shape[0], output.shape[1],
            bits, maxq,
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinearReal(nn.Module, TritonModuleMixin):
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
        if bits not in [2, 3, 4, 8]:
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


        
    def pack(self, linear, scales, zeros, g_idx=None):
        """
        Args:
            linear: nn.Linear allready quantized and dequantized
            scales: scales tensor of shape (infeatures//group_size, outfeatures)
            zeros: zeros tensor of shape (infeatures//group_size, outfeatures)
        """
        W = linear.weight.data.clone()
    
        g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32)

        scale_zeros = zeros * scales

        # 考虑有可能是bfloat16或者float16不要把参数限制的太死
        # import pdb; pdb.set_trace()
        scale_dtype = scales.dtype

        self.scales = nn.Parameter(scales.to(scale_dtype))
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(scale_dtype)

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
        # intweight = intweight.to(torch.uint32)

        i = 0
        row = 0
        # qweight (infeatures//(32//bits), outfeatures)
        # intweight (infeatures, outfeatures)
        qweight = np.zeros((math.ceil(intweight.shape[0]/(32//self.bits)),
                             intweight.shape[1]), dtype=np.uint32)
        
        # qweight = torch.zeros((math.ceil(intweight.shape[0]/(32//self.bits)),), dtype=torch.uint32)
        # RuntimeError: "lshift_cpu" not implemented for 'UInt32'
        while row < qweight.shape[0]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight).contiguous()
        # self.qweight = qweight.to(torch.uint32)
        zeros = zeros.float().numpy().astype(np.uint32)
        # zeros = zeros.to(torch.uint32)


        self.zeros_dim0, self.zeros_dim1 = zeros.shape
        # qzeros (infeatures//group_size, outfeatures//(32//bits))
        qzeros = np.zeros((zeros.shape[0], math.ceil(zeros.shape[1] / (32 // self.bits))), dtype=np.uint32)
        # qzeros = torch.zeros((zeros.shape[0], math.ceil(zeros.shape[1] / (32 // self.bits))), dtype=torch.uint32)


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
        self.qzeros = torch.from_numpy(qzeros).contiguous()
        # self.qzeros = qzeros.to(torch.uint32)


    def get_weight(self, transpose=False, dtype = torch.float16):
        weight = dequant_dim0(self.qweight, self.bits, self.maxq,
                               self.infeatures, self.outfeatures,dtype=dtype)
        dim0, dim1 = weight.shape
        zeros = dequant_dim1(self.qzeros, self.bits, self.maxq,
                              self.zeros_dim0, self.zeros_dim1, dtype=dtype)
        weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) 
            * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1)
        return weight.contiguous()
    
    def use_fake_quantization(self, del_quant=False,transpose=False, dtype = torch.float16):
        # use fake quantization for faster training but consume more memory
        weight = self.get_weight(transpose=transpose, dtype=dtype)
        # weight = dequant_dim0(self.qweight, self.bits,
        #          self.maxq, self.infeatures, self.outfeatures,dtype=dtype)
        # dim0, dim1 = weight.shape
        # zeros = dequant_dim1(self.qzeros, self.bits, self.maxq,
        #          self.zeros_dim0, self.zeros_dim1, dtype=dtype)
        # weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1))
        #      * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        # if transpose:
        #     self.fake_transpose = True
        #     weight = weight.transpose(0,1).contiguous()
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

    def _dequant_dim0(self):
        return dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)

    def _dequant_dim1(self):
        return dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)

    def forward(self, x):
        dtype = x.dtype
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            weight = self.get_weight(dtype=dtype)
        # import pdb; pdb.set_trace()
        out = torch.nn.functional.linear(x, weight.T.contiguous()) 
        # weight.contiguous()
        # out = x@weight
        return out,weight,x


import pytest

@torch.no_grad()
def init_fake_real_linear( linear:Linear):
    args= { }
    dtype = linear.weight.dtype
    linear_q_fake = QuantLinearFake(
        linear,
        wbits=2,
        group_size=128,
        args=args,
    )
    linear_q_fake.set_quant_state(True)
    # quant inplace 
    linear_q_fake.weight.data = linear_q_fake.weight_quantizer(
        linear_q_fake.weight.data
    )
    scales = linear_q_fake.weight_quantizer.scale.clamp(1e-4,1e4).clone()
    zeros = linear_q_fake.weight_quantizer.zero_point.clone()
    group_size = linear_q_fake.weight_quantizer.group_size
    dim0 = linear_q_fake.weight.shape[0]
    scales = scales.view(dim0,-1).transpose(0,1).contiguous()
    zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
    linear_q_real = QuantLinearReal(
        bits= 2,
        group_size=128,
        infeatures=linear.in_features,
        outfeatures=linear.out_features,
        bias=None,
    )
    linear_q_real.pack(linear_q_fake.cpu(), scales.cpu(), zeros.cpu())
    linear_q_fake.cuda(), linear_q_real.cuda()
    
    f_w = linear_q_fake.weight.data
    q_f_w = linear_q_fake.weight_quantizer(f_w)

    r_w = linear_q_real.get_weight(transpose=True,dtype=dtype)

    # # float16能够表达的最小正数是5.9e-8
    assert f_w.dtype == q_f_w.dtype == r_w.dtype == dtype , f"{f_w.dtype} {q_f_w.dtype} {r_w.dtype} {dtype}"
    assert torch.allclose(f_w, q_f_w, atol=1e-7)
    assert torch.allclose(q_f_w, r_w, atol=1e-7)
    assert torch.allclose(f_w, r_w, atol=1e-7)
    assert torch.eq(f_w,r_w).all()
    # # 保存为二进制文件
    # torch.save(f_w.contiguous(), "f_w.bin")
    # torch.save(r_w.contiguous(), "r_w.bin")

    # # 比较二进制文件内容
    # def compare_binary_files(file1, file2):
    #     with open(file1, "rb") as f1, open(file2, "rb") as f2:
    #         return f1.read() == f2.read()

    # if compare_binary_files("f_w.bin", "r_w.bin"):
    #     print("二进制文件一致")
    # else:
    #     print("二进制文件不一致")

    # # 转换为字节数据
    # f_w_bytes = f_w.cpu().numpy().tobytes()  # 先转 numpy，再转 bytes
    # r_w_bytes = r_w.cpu().numpy().tobytes()

    # # 比较字节数据
    # if f_w_bytes == r_w_bytes:
    #     print("二进制编码一致")
    # else:
    #     print("二进制编码不一致")
    
    return linear_q_fake, linear_q_real


@pytest.mark.parametrize("SIZE", [(4096, 14336),(4096,4096),(4096,11008),
                                  (896,4864),(896,896),(1536,8960),(1536,1536),
                                  (2048,11008),(2048,2048)]) # (in dim, out dim)
@pytest.mark.parametrize("BATCH", [2, 4])
@pytest.mark.parametrize("SEQLEN", [1024, 2048])
@pytest.mark.parametrize("DTYPE", [torch.float16])
@torch.no_grad()
def test_linear(SIZE, BATCH, SEQLEN,  DTYPE):
    rawlinear = Linear(SIZE[0], SIZE[1], False,dtype=DTYPE).cuda()
    init.kaiming_uniform_(rawlinear.weight, a=math.sqrt(5))
    # input_tensor = torch.randn([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    # 全0测试 pass
    # input_tensor = torch.zeros([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    # 全1测试 pass
    # input_tensor = torch.ones([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    # 单位矩阵 pass
    # input_tensor = torch.eye(SIZE[0], dtype=DTYPE, device="cuda")
    # 渐变输入
    input_tensor = torch.linspace(-5.5, 5.7, steps=BATCH * SEQLEN * SIZE[0],
         dtype=DTYPE, device="cuda").view(BATCH, SEQLEN, SIZE[0]).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    print(f"raw linear weight max {torch.max(rawlinear.weight)} min {torch.min(torch.abs(rawlinear.weight))}")
    print(f"input max {torch.max(input_tensor)} min {torch.min(input_tensor)}")

    linear_q_fake, linear_q_real = init_fake_real_linear(rawlinear)

    tensor_fake,w_fake,x_fake = linear_q_fake.forward(input_tensor)
    tensor_real,w_real,x_real = linear_q_real.forward(input_tensor)
    # import pdb; pdb.set_trace()
    linear_q_real.use_fake_quantization(dtype = DTYPE)
    tensor_us_fake,w_us_fake,x_us_fake = linear_q_real.forward(input_tensor)
    
    assert linear_q_fake.bias is None
    assert linear_q_real.bias is None

    assert (DTYPE == tensor_fake.dtype 
        == tensor_real.dtype == tensor_us_fake.dtype
        ),f"{DTYPE} {tensor_fake.dtype} {tensor_real.dtype} {tensor_us_fake.dtype}"
    atol = 1e-5
    print(f"amax = {torch.amax(tensor_real - tensor_fake)} norm = {torch.norm(tensor_real - tensor_fake)}")
    # 统计相差大于atol的个数
    print("real == fake?",torch.sum(torch.abs(tensor_real - tensor_fake) > atol))
    print("us fake == fake?",torch.sum(torch.abs(tensor_us_fake - tensor_fake) > atol))

    assert torch.allclose(tensor_fake, tensor_real,atol=atol)



if __name__ == "__main__":
    # test_linear((4096, 14336), 4, 2,  torch.bfloat16)
    # test_linear((896, 896), 4, 2048,  torch.bfloat16)
    # test_linear((4096, 14336), 2, 1024,  torch.float32)
    # test_linear((4096, 4096), 4, 2048,  torch.bfloat16)
    torch.manual_seed(0)
    test_linear((4096, 4096), 4, 2048,  torch.float16)