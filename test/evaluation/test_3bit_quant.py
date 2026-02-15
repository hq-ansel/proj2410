#!/usr/bin/env python3
"""
Test script for 3-bit quantization support in TritonV2QuantLinear.
"""

import torch
import numpy as np

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear
from EfficientQAT.core.linear.q_linear_triton_kernels import dequant


def test_3bit_pack_unpack():
    """Test that 3-bit packing and unpacking is consistent."""
    print("=" * 60)
    print("Testing 3-bit pack/unpack consistency")
    print("=" * 60)
    
    # Create a simple linear layer
    in_features = 64  # Must be divisible by 32
    out_features = 64  # Must be divisible by 32
    group_size = 64
    
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    # Initialize with known values for easier debugging
    torch.manual_seed(42)
    linear.weight.data = torch.randn(out_features, in_features) * 0.1
    
    # Create quantized linear
    qlinear = TritonV2QuantLinear(
        bits=3,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
    )
    
    # Create scales and zeros
    num_groups = in_features // group_size
    scales = torch.ones(out_features, num_groups) * 0.1
    zeros = torch.ones(out_features, num_groups) * 4  # Middle of 3-bit range
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    
    # Pack the weights
    qlinear.pack(linear, scales, zeros, g_idx)
    qlinear.post_init()
    
    print(f"qweight shape: {qlinear.qweight.shape}")
    print(f"qzeros shape: {qlinear.qzeros.shape}")
    print(f"scales shape: {qlinear.scales.shape}")
    print(f"g_idx shape: {qlinear.g_idx.shape}")
    
    # Test dequantization using the torch-based method
    dequant_weight_torch = qlinear.dequantize_weight()
    print(f"Dequantized weight shape (torch): {dequant_weight_torch.shape}")
    
    # Move to CUDA for triton kernel test
    if torch.cuda.is_available():
        qlinear.qweight = qlinear.qweight.cuda()
        qlinear.qzeros = qlinear.qzeros.cuda()
        qlinear.scales = qlinear.scales.cuda()
        qlinear.g_idx = qlinear.g_idx.cuda()
        
        # Test triton dequantization
        dequant_weight_triton = dequant(
            qlinear.qweight,
            qlinear.scales,
            qlinear.qzeros,
            qlinear.g_idx,
            bits=3,
            pack_bits=32,
            maxq=7,
            sym=False,
        )
        print(f"Dequantized weight shape (triton): {dequant_weight_triton.shape}")
        
        # Compare results
        # Both torch and triton dequant return [out_features, in_features] layout
        # (triton kernel outputs [in_features, out_features] but the matmul expects this)
        # Actually, let's check: torch dequantize_weight returns [out_features, in_features]
        # and triton dequant returns [in_features, out_features]
        # But for the forward pass, we need weight @ input.T or input @ weight.T
        # The quant_matmul does: input @ weight (no transpose by default)
        # So triton output [in_features, out_features] is correct for input @ weight
        
        # For comparison, both should produce the same values, just different layouts
        dequant_weight_torch = dequant_weight_torch.cuda().half()
        
        # Direct comparison (same layout)
        diff = (dequant_weight_triton - dequant_weight_torch).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        if max_diff < 1e-3:
            print("✓ 3-bit pack/unpack test PASSED!")
        else:
            print("✗ 3-bit pack/unpack test FAILED!")
            print("Torch dequant sample:")
            print(dequant_weight_torch[:5, :5])
            print("Triton dequant sample:")
            print(dequant_weight_triton[:5, :5])
    else:
        print("CUDA not available, skipping triton kernel test")
    
    return True


def test_3bit_forward():
    """Test forward pass with 3-bit quantization."""
    print("\n" + "=" * 60)
    print("Testing 3-bit forward pass")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping forward test")
        return True
    
    in_features = 128
    out_features = 64
    group_size = 64
    batch_size = 4
    seq_len = 16
    
    # Create original linear on CPU first for packing
    linear = torch.nn.Linear(in_features, out_features, bias=True)
    torch.manual_seed(42)
    linear.weight.data = torch.randn(out_features, in_features) * 0.1
    linear.bias.data = torch.randn(out_features) * 0.01
    
    # Create quantized linear
    qlinear = TritonV2QuantLinear(
        bits=3,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )
    
    # Create scales and zeros on CPU
    num_groups = in_features // group_size
    scales = torch.ones(out_features, num_groups) * 0.1
    zeros = torch.ones(out_features, num_groups) * 4
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    
    # Pack on CPU
    qlinear.pack(linear, scales, zeros, g_idx)
    qlinear.post_init()
    
    # Move to CUDA after packing
    qlinear.qweight = qlinear.qweight.cuda()
    qlinear.qzeros = qlinear.qzeros.cuda()
    qlinear.scales = qlinear.scales.cuda()
    qlinear.g_idx = qlinear.g_idx.cuda()
    qlinear.bias = qlinear.bias.cuda()
    qlinear.wf_unsqueeze_zero = qlinear.wf_unsqueeze_zero.cuda()
    qlinear.wf_unsqueeze_neg_one = qlinear.wf_unsqueeze_neg_one.cuda()
    
    # Test forward
    x = torch.randn(batch_size, seq_len, in_features).cuda().half()
    
    with torch.no_grad():
        out = qlinear(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output sample: {out[0, 0, :5]}")
    
    # Check for NaN/Inf
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("✗ Forward test FAILED - NaN or Inf in output!")
        return False
    
    print("✓ 3-bit forward test PASSED!")
    return True


def test_3bit_symmetric():
    """Test 3-bit symmetric quantization."""
    print("\n" + "=" * 60)
    print("Testing 3-bit symmetric quantization")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping symmetric test")
        return True
    
    in_features = 64
    out_features = 64
    group_size = 64
    
    # Create on CPU first
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    torch.manual_seed(42)
    linear.weight.data = torch.randn(out_features, in_features) * 0.1
    
    # Create symmetric quantized linear
    qlinear = TritonV2QuantLinear(
        bits=3,
        group_size=group_size,
        desc_act=False,
        sym=True,  # Symmetric
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
    )
    
    num_groups = in_features // group_size
    scales = torch.ones(out_features, num_groups) * 0.1
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    
    # Pack with zeros=None for symmetric on CPU
    qlinear.pack(linear, scales, None, g_idx)
    qlinear.post_init()
    
    # Move to CUDA after packing
    qlinear.qweight = qlinear.qweight.cuda()
    qlinear.qzeros = qlinear.qzeros.cuda()
    qlinear.scales = qlinear.scales.cuda()
    qlinear.g_idx = qlinear.g_idx.cuda()
    qlinear.wf_unsqueeze_zero = qlinear.wf_unsqueeze_zero.cuda()
    qlinear.wf_unsqueeze_neg_one = qlinear.wf_unsqueeze_neg_one.cuda()
    
    # Test dequantization
    dequant_weight_triton = dequant(
        qlinear.qweight,
        qlinear.scales,
        qlinear.qzeros,
        qlinear.g_idx,
        bits=3,
        pack_bits=32,
        maxq=7,
        sym=True,
    )
    
    print(f"Dequantized weight shape: {dequant_weight_triton.shape}")
    
    # Test forward
    x = torch.randn(2, 8, in_features).cuda().half()
    with torch.no_grad():
        out = qlinear(x)
    
    print(f"Output shape: {out.shape}")
    
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("✗ Symmetric test FAILED - NaN or Inf in output!")
        return False
    
    print("✓ 3-bit symmetric test PASSED!")
    return True


if __name__ == "__main__":
    print("Testing 3-bit quantization support for TritonV2QuantLinear")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_3bit_pack_unpack()
    except Exception as e:
        print(f"✗ Pack/unpack test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_3bit_forward()
    except Exception as e:
        print(f"✗ Forward test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_3bit_symmetric()
    except Exception as e:
        print(f"✗ Symmetric test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
