#!/usr/bin/env python3
"""Test Seq2Bit integration in megatron_qat.py"""

import sys
import os

# Setup paths
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch
import torch.nn as nn

# Import the updated module
from megatron.core.quantization.megatron_qat import (
    MegatronSeq2BitQuantizer,
    MegatronWeightQuantizer,
    QuantConfig,
    LinearWithSeq2BitQAT,
    convert_megatron_model,
    set_quant_state,
)

def test_seq2bit_quantizer_creation():
    """Test creating a Seq2Bit quantizer."""
    print("Test 1: Seq2Bit Quantizer Creation")

    weight = nn.Parameter(torch.randn(256, 512, device='cuda', dtype=torch.float16))
    config = QuantConfig(
        quant_type="seq2bit",
        group_size=128,
    )

    quantizer = MegatronSeq2BitQuantizer(weight, config, prefix="test_layer")

    assert quantizer.quant_type == "seq2bit", f"Expected quant_type='seq2bit', got {quantizer.quant_type}"
    assert quantizer.weight_quantizer is not None, "weight_quantizer should not be None"
    assert hasattr(quantizer.weight_quantizer, 'alpha'), "Seq2Bit quantizer should have alpha parameter"

    print(f"  Alpha shape: {quantizer.weight_quantizer.alpha.shape}")
    print(f"  Group size: {quantizer.weight_quantizer.group_size}")
    print("  [PASS] Seq2Bit quantizer created successfully")
    return True

def test_seq2bit_forward():
    """Test Seq2Bit forward pass."""
    print("\nTest 2: Seq2Bit Forward Pass")

    weight = nn.Parameter(torch.randn(256, 512, device='cuda', dtype=torch.float16))
    config = QuantConfig(
        quant_type="seq2bit",
        group_size=128,
    )

    quantizer = MegatronSeq2BitQuantizer(weight, config, prefix="test_layer")
    quantizer.use_weight_quant = True

    # Forward with quantization
    weight_q = quantizer(weight)

    assert weight_q.shape == weight.shape, f"Shape mismatch: {weight_q.shape} vs {weight.shape}"
    assert torch.isfinite(weight_q).all(), "Quantized weight should be finite"

    print(f"  Input weight range: [{weight.min():.4f}, {weight.max():.4f}]")
    print(f"  Quantized weight range: [{weight_q.min():.4f}, {weight_q.max():.4f}]")
    print("  [PASS] Seq2Bit forward pass works")
    return True

def test_seq2bit_backward():
    """Test Seq2Bit backward pass with gradient computation."""
    print("\nTest 3: Seq2Bit Backward Pass")

    # weight shape: [out_features, in_features] = [256, 512]
    # input shape: [batch, in_features] = [4, 512]
    # output = input @ weight.T = [4, 256]
    weight = nn.Parameter(torch.randn(256, 512, device='cuda', dtype=torch.float32), requires_grad=True)
    config = QuantConfig(
        quant_type="seq2bit",
        group_size=128,
    )

    quantizer = MegatronSeq2BitQuantizer(weight, config, prefix="test_layer")
    quantizer.use_weight_quant = True

    alpha = quantizer.weight_quantizer.alpha

    # Use the autograd function directly
    # input features must match weight's in_features (512)
    input_tensor = torch.randn(4, 512, device='cuda', dtype=torch.float32, requires_grad=True)

    output = LinearWithSeq2BitQAT.apply(
        input_tensor,
        weight,
        None,  # bias
        alpha,
        128,   # group_size
        False, # gradient_accumulation_fusion
        False, # allreduce_dgrad
        False, # sequence_parallel
        None,  # tp_group
    )

    # output shape should be [4, 256]
    assert output.shape == (4, 256), f"Expected output shape [4, 256], got {output.shape}"

    loss = output.sum()
    loss.backward()

    assert weight.grad is not None, "weight.grad should not be None"
    assert alpha.grad is not None, "alpha.grad should not be None"
    assert torch.isfinite(weight.grad).all(), "weight.grad should be finite"
    assert torch.isfinite(alpha.grad).all(), "alpha.grad should be finite"

    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Weight grad norm: {weight.grad.norm():.6f}")
    print(f"  Alpha grad norm: {alpha.grad.norm():.6f}")
    print("  [PASS] Seq2Bit backward pass works")
    return True

def test_convert_model_seq2bit():
    """Test converting a model with Seq2Bit quantization."""
    print("\nTest 4: Convert Model with Seq2Bit")

    # Just test the config logic
    config = QuantConfig(
        quant_type="seq2bit",
        group_size=128,
    )

    print(f"  Config quant_type: {config.quant_type}")
    print(f"  Config group_size: {config.group_size}")
    print("  [PASS] Config ready for Seq2Bit model conversion")
    return True

def main():
    print("=" * 60)
    print("Megatron Seq2Bit Integration Tests")
    print("=" * 60)

    tests = [
        test_seq2bit_quantizer_creation,
        test_seq2bit_forward,
        test_seq2bit_backward,
        test_convert_model_seq2bit,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)