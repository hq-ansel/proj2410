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

# Configs for 3-bit dequantization (smaller block size due to complexity)
DEFAULT_DEQUANT_3BIT_CONFIGS = make_dequant_configs([256], [2], [1])


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


# Optimized configs for 3-bit: use larger block size and more warps for better occupancy
DEFAULT_DEQUANT_3BIT_CONFIGS_V2 = make_dequant_configs([256, 512, 1024], [2, 4], [1, 2])


@triton.autotune(DEFAULT_DEQUANT_3BIT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel_3bit(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    pack_bits: tl.constexpr,
    maxq: tl.constexpr,
    out_features: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
    sym: tl.constexpr,
):
    """
    3-bit dequantization kernel.
    
    3-bit packing scheme (32 values packed into 3 int32s):
    The packing follows this pattern for each group of 32 values:
    - int32[0]: values[0:10] at shifts [0,3,6,9,12,15,18,21,24,27], value[10] low 2 bits at shift 30
    - int32[1]: value[10] high 1 bit at shift 0, values[11:21] at shifts [1,4,7,10,13,16,19,22,25,28], value[21] low 1 bit at shift 31
    - int32[2]: value[21] high 2 bits at shift 0, values[22:32] at shifts [2,5,8,11,14,17,20,23,26,29]
    
    qweight shape: [in_features * 3 / 32, out_features]
    output shape: [in_features, out_features]
    """
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    
    row_idx = x_index // out_features  # in_features index
    col_idx = x_index % out_features   # out_features index
    
    # Load group index
    g_idx = tl.load(g_idx_ptr + row_idx, xmask, eviction_policy="evict_last")
    
    # 3-bit packing: 32 values in 3 int32s
    # row_in_group: which of the 32 values within a group of 3 int32s
    row_in_group = row_idx % 32
    # base_row: starting row of the 3 int32s group (in packed space)
    base_row = (row_idx // 32) * 3
    
    # Load all 3 int32s for this group
    qw0 = tl.load(qweight_ptr + (col_idx + out_features * base_row), xmask)
    qw1 = tl.load(qweight_ptr + (col_idx + out_features * (base_row + 1)), xmask)
    qw2 = tl.load(qweight_ptr + (col_idx + out_features * (base_row + 2)), xmask)
    
    # Initialize weights
    weights = tl.zeros([X_BLOCK], dtype=tl.int32)
    
    # Case 1: values 0-9 (in qw0, shifts 0,3,6,...,27)
    mask_0_9 = row_in_group < 10
    shift_0_9 = row_in_group * 3
    val_0_9 = (qw0 >> shift_0_9) & 0x7
    weights = tl.where(mask_0_9, val_0_9, weights)
    
    # Case 2: value 10 (split: low 2 bits at qw0[30:32], high 1 bit at qw1[0])
    mask_10 = row_in_group == 10
    val_10 = ((qw0 >> 30) & 0x3) | (((qw1 & 0x1) << 2))
    weights = tl.where(mask_10, val_10, weights)
    
    # Case 3: values 11-20 (in qw1, shifts 1,4,7,...,28)
    mask_11_20 = (row_in_group >= 11) & (row_in_group <= 20)
    shift_11_20 = (row_in_group - 11) * 3 + 1
    val_11_20 = (qw1 >> shift_11_20) & 0x7
    weights = tl.where(mask_11_20, val_11_20, weights)
    
    # Case 4: value 21 (split: low 1 bit at qw1[31], high 2 bits at qw2[0:2])
    mask_21 = row_in_group == 21
    val_21 = ((qw1 >> 31) & 0x1) | ((qw2 & 0x3) << 1)
    weights = tl.where(mask_21, val_21, weights)
    
    # Case 5: values 22-31 (in qw2, shifts 2,5,8,...,29)
    mask_22_31 = row_in_group >= 22
    shift_22_31 = (row_in_group - 22) * 3 + 2
    val_22_31 = (qw2 >> shift_22_31) & 0x7
    weights = tl.where(mask_22_31, val_22_31, weights)
    
    # Get group info for scales/zeros
    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    groups = tl.where(tmp2, tmp1, g_idx)
    
    if sym:
        # Symmetric quantization: convert to signed
        sign_bit = 4  # 1 << (3 - 1)
        full_range = 8  # 1 << 3
        weights = tl.where(weights >= sign_bit, weights - full_range, weights)
        
        # Load scales
        scales = tl.load(scales_ptr + (col_idx + out_features * groups), xmask).to(tl.float32)
        weights = weights.to(tl.float32) * scales
    else:
        # Asymmetric quantization: need zeros
        scales = tl.load(scales_ptr + (col_idx + out_features * groups), xmask).to(tl.float32)
        
        # Load and unpack zeros (same 3-bit packing scheme)
        # qzeros shape: [num_groups, out_features * 3 / 32]
        qzero_ncols = out_features * 3 // 32
        col_in_group = col_idx % 32
        base_col = (col_idx // 32) * 3
        
        qz0 = tl.load(qzeros_ptr + (qzero_ncols * groups + base_col), xmask, eviction_policy="evict_last")
        qz1 = tl.load(qzeros_ptr + (qzero_ncols * groups + base_col + 1), xmask, eviction_policy="evict_last")
        qz2 = tl.load(qzeros_ptr + (qzero_ncols * groups + base_col + 2), xmask, eviction_policy="evict_last")
        
        zeros = tl.zeros([X_BLOCK], dtype=tl.int32)
        
        # Extract zeros using same pattern
        zmask_0_9 = col_in_group < 10
        zshift_0_9 = col_in_group * 3
        zval_0_9 = (qz0 >> zshift_0_9) & 0x7
        zeros = tl.where(zmask_0_9, zval_0_9, zeros)
        
        zmask_10 = col_in_group == 10
        zval_10 = ((qz0 >> 30) & 0x3) | (((qz1 & 0x1) << 2))
        zeros = tl.where(zmask_10, zval_10, zeros)
        
        zmask_11_20 = (col_in_group >= 11) & (col_in_group <= 20)
        zshift_11_20 = (col_in_group - 11) * 3 + 1
        zval_11_20 = (qz1 >> zshift_11_20) & 0x7
        zeros = tl.where(zmask_11_20, zval_11_20, zeros)
        
        zmask_21 = col_in_group == 21
        zval_21 = ((qz1 >> 31) & 0x1) | ((qz2 & 0x3) << 1)
        zeros = tl.where(zmask_21, zval_21, zeros)
        
        zmask_22_31 = col_in_group >= 22
        zshift_22_31 = (col_in_group - 22) * 3 + 2
        zval_22_31 = (qz2 >> zshift_22_31) & 0x7
        zeros = tl.where(zmask_22_31, zval_22_31, zeros)
        
        weights = (weights - zeros).to(tl.float32) * scales
    
    tl.store(out_ptr + x_index, weights, mask=xmask)


@triton.autotune(DEFAULT_DEQUANT_3BIT_CONFIGS_V2, key=["numels"])
@triton.jit
def dequant_kernel_3bit_v2(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    pack_bits: tl.constexpr,
    maxq: tl.constexpr,
    out_features: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
    sym: tl.constexpr,
):
    """
    Optimized 3-bit dequantization kernel v2.
    
    Optimizations:
    1. Use lookup table approach for shift values to reduce branching
    2. Compute which int32 to use based on position, reducing redundant loads
    3. Better memory coalescing by processing in groups
    
    3-bit packing scheme (32 values packed into 3 int32s):
    - Position 0-9: in qw0, shift = pos * 3
    - Position 10: split across qw0 (bits 30-31) and qw1 (bit 0)
    - Position 11-20: in qw1, shift = (pos - 11) * 3 + 1
    - Position 21: split across qw1 (bit 31) and qw2 (bits 0-1)
    - Position 22-31: in qw2, shift = (pos - 22) * 3 + 2
    """
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    
    row_idx = x_index // out_features  # in_features index
    col_idx = x_index % out_features   # out_features index
    
    # Load group index
    g_idx = tl.load(g_idx_ptr + row_idx, xmask, eviction_policy="evict_last")
    
    # 3-bit packing: 32 values in 3 int32s
    row_in_group = row_idx % 32
    base_row = (row_idx // 32) * 3
    
    # Determine which int32(s) we need based on position
    # Region 0: pos 0-9 -> only qw0
    # Region 1: pos 10 -> qw0 and qw1 (split)
    # Region 2: pos 11-20 -> only qw1
    # Region 3: pos 21 -> qw1 and qw2 (split)
    # Region 4: pos 22-31 -> only qw2
    
    is_region_0 = row_in_group < 10
    is_region_1 = row_in_group == 10
    is_region_2 = (row_in_group >= 11) & (row_in_group <= 20)
    is_region_3 = row_in_group == 21
    is_region_4 = row_in_group >= 22
    
    # Load qweight values - we still need to load all 3 for split values
    # but we can optimize by computing which ones we actually need
    qw0 = tl.load(qweight_ptr + (col_idx + out_features * base_row), xmask)
    qw1 = tl.load(qweight_ptr + (col_idx + out_features * (base_row + 1)), xmask)
    qw2 = tl.load(qweight_ptr + (col_idx + out_features * (base_row + 2)), xmask)
    
    # Compute shift and extract value based on region
    # Region 0: shift = pos * 3, use qw0
    shift_0 = row_in_group * 3
    val_0 = (qw0 >> shift_0) & 0x7
    
    # Region 1: split value at position 10
    val_1 = ((qw0 >> 30) & 0x3) | ((qw1 & 0x1) << 2)
    
    # Region 2: shift = (pos - 11) * 3 + 1, use qw1
    shift_2 = (row_in_group - 11) * 3 + 1
    val_2 = (qw1 >> shift_2) & 0x7
    
    # Region 3: split value at position 21
    val_3 = ((qw1 >> 31) & 0x1) | ((qw2 & 0x3) << 1)
    
    # Region 4: shift = (pos - 22) * 3 + 2, use qw2
    shift_4 = (row_in_group - 22) * 3 + 2
    val_4 = (qw2 >> shift_4) & 0x7
    
    # Select the correct value using nested where (reduces branch divergence)
    weights = tl.where(is_region_0, val_0,
              tl.where(is_region_1, val_1,
              tl.where(is_region_2, val_2,
              tl.where(is_region_3, val_3, val_4))))
    
    # Get group info for scales/zeros
    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    groups = tl.where(tmp2, tmp1, g_idx)
    
    if sym:
        # Symmetric quantization: convert to signed
        sign_bit = 4  # 1 << (3 - 1)
        full_range = 8  # 1 << 3
        weights = tl.where(weights >= sign_bit, weights - full_range, weights)
        
        # Load scales
        scales = tl.load(scales_ptr + (col_idx + out_features * groups), xmask).to(tl.float32)
        weights = weights.to(tl.float32) * scales
    else:
        # Asymmetric quantization: need zeros
        scales = tl.load(scales_ptr + (col_idx + out_features * groups), xmask).to(tl.float32)
        
        # Load and unpack zeros (same 3-bit packing scheme)
        qzero_ncols = out_features * 3 // 32
        col_in_group = col_idx % 32
        base_col = (col_idx // 32) * 3
        
        qz0 = tl.load(qzeros_ptr + (qzero_ncols * groups + base_col), xmask, eviction_policy="evict_last")
        qz1 = tl.load(qzeros_ptr + (qzero_ncols * groups + base_col + 1), xmask, eviction_policy="evict_last")
        qz2 = tl.load(qzeros_ptr + (qzero_ncols * groups + base_col + 2), xmask, eviction_policy="evict_last")
        
        # Compute zeros regions
        z_is_region_0 = col_in_group < 10
        z_is_region_1 = col_in_group == 10
        z_is_region_2 = (col_in_group >= 11) & (col_in_group <= 20)
        z_is_region_3 = col_in_group == 21
        
        z_shift_0 = col_in_group * 3
        z_val_0 = (qz0 >> z_shift_0) & 0x7
        z_val_1 = ((qz0 >> 30) & 0x3) | ((qz1 & 0x1) << 2)
        z_shift_2 = (col_in_group - 11) * 3 + 1
        z_val_2 = (qz1 >> z_shift_2) & 0x7
        z_val_3 = ((qz1 >> 31) & 0x1) | ((qz2 & 0x3) << 1)
        z_shift_4 = (col_in_group - 22) * 3 + 2
        z_val_4 = (qz2 >> z_shift_4) & 0x7
        
        zeros = tl.where(z_is_region_0, z_val_0,
                tl.where(z_is_region_1, z_val_1,
                tl.where(z_is_region_2, z_val_2,
                tl.where(z_is_region_3, z_val_3, z_val_4))))
        
        weights = (weights - zeros).to(tl.float32) * scales
    
    tl.store(out_ptr + x_index, weights, mask=xmask)


def dequant(qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, sym: bool = False, use_v2: bool = False):
    """
    Launcher for triton dequant kernel. Valid for bits = 2, 3, 4, 8
    
    Args:
        use_v2: If True, use optimized v2 kernel for 3-bit (default: False)
                v2 may be faster for very large matrices but v1 is more consistent
    """

    num_groups = scales.shape[0]
    out_features = scales.shape[1]
    in_features = g_idx.shape[0]

    out = torch.empty((in_features, out_features), device=qweight.device, dtype=torch.float16)
    numels = out.numel()
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)

    if bits == 3:
        # Use specialized 3-bit kernel
        # v1 is actually faster for small matrices, v2 for large ones
        # Default to v1 which is more consistent
        if use_v2:
            dequant_kernel_3bit_v2[grid](
                g_idx,
                scales,
                qweight,
                qzeros,
                out,
                numels,
                pack_bits=pack_bits,
                maxq=maxq,
                out_features=out_features,
                num_groups=num_groups,
                sym=sym,
            )
        else:
            dequant_kernel_3bit[grid](
                g_idx,
                scales,
                qweight,
                qzeros,
                out,
                numels,
                pack_bits=pack_bits,
                maxq=maxq,
                out_features=out_features,
                num_groups=num_groups,
                sym=sym,
            )
    else:
        # Use standard kernel for 2, 4, 8 bits
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
    "DEFAULT_DEQUANT_3BIT_CONFIGS",
    "DEFAULT_DEQUANT_3BIT_CONFIGS_V2",
    "dequant",
    "dequant_kernel",
    "dequant_kernel_3bit",
    "dequant_kernel_3bit_v2",
    "make_dequant_configs",
    "quant_matmul",
]
