import itertools

import torch
import triton
import triton.language as tl


def make_dequant_configs(block_sizes, num_warps, num_stages):
    configs = []
    for bs, ws, ns in itertools.product(block_sizes, num_warps, num_stages):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws, num_stages=ns))
    return configs


# tested on A100 with [Llama 3.2 1B and Falcon 7B] bits:4, group_size:128
DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([512], [1], [1])


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    pack_bits: tl.constexpr,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    out_features: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
    sym: tl.constexpr,
):
    # Block indexing
    """
    Dequantizes packed quantized weights using Triton kernel.
    
    Args:
        g_idx_ptr: Pointer to group indices tensor
        scales_ptr: Pointer to scales tensor
        qweight_ptr: Pointer to quantized weights tensor
        qzeros_ptr: Pointer to quantized zeros tensor  
        out_ptr: Pointer to output tensor for dequantized values
        numels: Total number of elements to process
        pack_bits: Number of bits used for packing (constexpr)
        maxq: Maximum quantization value (constexpr)
        bits: Number of bits per quantized value (constexpr)
        out_features: Number of output features (constexpr)
        num_groups: Number of quantization groups (constexpr)
        X_BLOCK: Block size for parallel processing (constexpr)
    
    Process:
        1. Loads and unpacks quantized weights and zeros
        2. Adjusts for group indices
        3. Applies dequantization formula: (weights - zeros) * scales
        4. Stores dequantized values to output tensor
    """
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // out_features
    col_idx = x_index % out_features

    elements_per_feature: tl.constexpr = pack_bits // bits

    # Load parameters
    g_idx = tl.load(g_idx_ptr + (row_idx), None, eviction_policy="evict_last")
    qweights = tl.load(
        qweight_ptr + (col_idx + (out_features * (row_idx // elements_per_feature))),
        None,
    )

    wf_weights = (row_idx % elements_per_feature) * bits
    wf_zeros = (col_idx % elements_per_feature) * bits

    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    groups = tl.where(tmp2, tmp1, g_idx)

    scales = tl.load(scales_ptr + (col_idx + (out_features * groups)), None).to(tl.float32)

    # Unpack weights
    weights = (qweights >> wf_weights) & maxq  # bit shift qweight

    if sym:
        sign_bit = (1 << (bits - 1))
        full_range = (1 << bits)
        weights = tl.where(weights >= sign_bit, weights - full_range, weights)
        weights = weights.to(tl.float32) * scales
    else:
        # Unpack zeros
        qzero_ncols: tl.constexpr = out_features // elements_per_feature
        qzeros = tl.load(
            qzeros_ptr + ((qzero_ncols * groups) + (col_idx // elements_per_feature)),
            None,
            eviction_policy="evict_last",
        )
        zeros = (qzeros >> wf_zeros) & maxq
        # Dequantize
        weights = (weights - zeros).to(tl.float32) * scales

    tl.store(out_ptr + (x_index), weights, mask=xmask)


def dequant(qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, sym: bool = False):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """

    num_groups = scales.shape[0]
    out_features = scales.shape[1]
    in_features = g_idx.shape[0]

    out = torch.empty((in_features, out_features), device=qweight.device, dtype=torch.float16)
    numels = out.numel()
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)

    dequant_kernel[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        out,
        numels,
        pack_bits=pack_bits,
        maxq=maxq,
        bits=bits,
        out_features=out_features,
        num_groups=num_groups,
        sym=sym,
    )
    return out


def quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=False, sym: bool = False):
    weight = dequant(qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, sym=sym)
    if transpose:
        return input @ weight.t()
    return input @ weight


__all__ = [
    "DEFAULT_DEQUANT_CONFIGS",
    "dequant",
    "dequant_kernel",
    "make_dequant_configs",
    "quant_matmul",
]
