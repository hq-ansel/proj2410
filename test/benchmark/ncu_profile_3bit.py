#!/usr/bin/env python3
"""
NCU profiling script for 3-bit dequantization kernel.
Run with: ncu --set full -o tmp/ncu_3bit_dequant python test/ncu_profile_3bit.py
Or for quick analysis: ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,dram__throughput.avg_pct_of_peak_sustained_elapsed python test/ncu_profile_3bit.py
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear
from EfficientQAT.core.linear.q_linear_triton_kernels import dequant


def profile_dequant(bits, in_features, out_features, group_size, use_v2=False):
    """Profile dequantization kernel."""
    
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    torch.manual_seed(42)
    linear.weight.data = torch.randn(out_features, in_features) * 0.1
    
    qlinear = TritonV2QuantLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
    )
    
    num_groups = in_features // group_size
    scales = torch.ones(out_features, num_groups) * 0.1
    zeros = torch.ones(out_features, num_groups) * (2 ** (bits - 1))
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    
    qlinear.pack(linear, scales, zeros, g_idx)
    qlinear.post_init()
    
    # Move to CUDA
    qlinear.qweight = qlinear.qweight.cuda()
    qlinear.qzeros = qlinear.qzeros.cuda()
    qlinear.scales = qlinear.scales.cuda()
    qlinear.g_idx = qlinear.g_idx.cuda()
    
    # Warmup
    for _ in range(3):
        _ = dequant(
            qlinear.qweight,
            qlinear.scales,
            qlinear.qzeros,
            qlinear.g_idx,
            bits=bits,
            pack_bits=32,
            maxq=2**bits - 1,
            sym=False,
            use_v2=use_v2,
        )
    torch.cuda.synchronize()
    
    # Profile this call
    version = "v2" if use_v2 else "v1"
    print(f"Profiling {bits}-bit ({version}) dequant: {in_features}x{out_features}, group_size={group_size}")
    result = dequant(
        qlinear.qweight,
        qlinear.scales,
        qlinear.qzeros,
        qlinear.g_idx,
        bits=bits,
        pack_bits=32,
        maxq=2**bits - 1,
        sym=False,
        use_v2=use_v2,
    )
    torch.cuda.synchronize()
    
    return result


if __name__ == "__main__":
    # Profile 3-bit kernel with typical LLM sizes
    in_features = 4096
    out_features = 4096
    group_size = 128
    
    # Profile 4-bit for comparison
    print("=" * 60)
    print("Profiling 4-bit kernel (baseline)")
    print("=" * 60)
    _ = profile_dequant(4, in_features, out_features, group_size)
    
    print()
    print("=" * 60)
    print("Profiling 3-bit kernel (v1)")
    print("=" * 60)
    _ = profile_dequant(3, in_features, out_features, group_size, use_v2=False)
    
    print()
    print("=" * 60)
    print("Profiling 3-bit kernel (v2)")
    print("=" * 60)
    _ = profile_dequant(3, in_features, out_features, group_size, use_v2=True)
    
    print("\nDone. Check NCU report for details.")
    print("\nKey metrics to look at:")
    print("- SM Throughput: How well the compute units are utilized")
    print("- Memory Throughput: How well memory bandwidth is utilized")
    print("- Warp Stall Reasons: What's causing threads to wait")
    print("- Branch Efficiency: Impact of conditional branches")
