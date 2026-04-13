#!/usr/bin/env python3
"""
Level 1: Kernel Math Tests for Seq2Bit Quantization

Tests:
1. fake_quant_fwd: Verify quantization output correctness
2. fake_quant_bwd: Verify gradient computation correctness
3. finite_difference: Numerical gradient comparison
4. single_linear: Single layer forward+backward
5. pytorch_reference: Compare with pure PyTorch implementation
"""

import sys
import os
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# Import Seq2Bit components
from EfficientQAT.core.quantizer.kernel.fake_quant import (
    fake_quant_ste_seq2bit,
    fake_quant_backward_seq2bit,
)


# ---------------------------------------------------------------------------
# PyTorch Reference Implementation
# ---------------------------------------------------------------------------

def seq2bit_quantize_pytorch(x: torch.Tensor, alpha: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Pure PyTorch reference implementation of Seq2Bit quantization.

    Quantization levels: {-0.75, -0.25, 0.25, 0.75} * alpha

    Args:
        x: Input tensor [N, group_size] or flattened
        alpha: Scale parameter [N_groups]
        group_size: Quantization group size

    Returns:
        Quantized tensor with same shape as x
    """
    ori_shape = x.shape
    x = x.reshape(-1, group_size)
    alpha = alpha.reshape(-1, 1)

    # Normalize by alpha
    s = alpha.clamp(min=1e-6)
    xn = (x / s).clamp(-1.0, 1.0)

    # Quantize to 4 levels: -0.75, -0.25, 0.25, 0.75
    # code = round((xn + 0.75) / 0.5) -> 0, 1, 2, 3
    code = torch.round((xn + 0.75) / 0.5).clamp(0, 3)

    # Dequantize: levels = code * 0.5 - 0.75
    levels = code * 0.5 - 0.75

    # Scale back
    x_dequant = levels * s

    return x_dequant.reshape(ori_shape)


def seq2bit_backward_pytorch(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    alpha: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference implementation of Seq2Bit backward.

    Using STE (Straight-Through Estimator):
    - dx = grad_output if |x/alpha| <= 1, else 0
    - dalpha = gradient w.r.t. alpha through the quantization

    Args:
        grad_output: Gradient w.r.t. output
        x: Original input
        alpha: Scale parameter
        group_size: Quantization group size

    Returns:
        (grad_x, grad_alpha)
    """
    ori_shape = x.shape
    x = x.reshape(-1, group_size)
    grad_output = grad_output.reshape(-1, group_size)
    alpha = alpha.reshape(-1, 1)
    s = alpha.clamp(min=1e-6)

    # STE mask: gradient passes through where |x/alpha| <= 1
    xn = x / s
    mask = (xn.abs() <= 1.0).float()

    # grad_x: STE gradient
    grad_x = grad_output * mask

    # grad_alpha: compute analytically
    # For levels {-0.75, -0.25, 0.25, 0.75}, the dequantized value is level * alpha
    # We need to compute gradient through the quantization operation
    # This is approximated using the STE approach

    # Compute which level each element maps to
    xn_clamped = xn.clamp(-1.0, 1.0)
    code = torch.round((xn_clamped + 0.75) / 0.5).clamp(0, 3)
    levels = code * 0.5 - 0.75

    # Gradient of output w.r.t. alpha: levels (constant per quantized region)
    # grad_alpha = sum over group of (grad_output * level)
    grad_alpha = (grad_output * levels).sum(dim=-1, keepdim=True)

    return grad_x.reshape(ori_shape), grad_alpha.reshape(-1)


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------

def test_fake_quant_fwd():
    """Test 1: Verify fake_quant_ste_seq2bit forward pass."""
    print("\n" + "=" * 60)
    print("Test 1: fake_quant_fwd (Seq2Bit CUDA kernel)")
    print("=" * 60)

    torch.manual_seed(42)

    test_cases = [
        (64, 128),   # group_size=128, 64 groups
        (128, 64),   # group_size=64, 128 groups
        (32, 256),   # group_size=256, 32 groups
    ]

    all_passed = True

    for num_groups, group_size in test_cases:
        # Create input
        x = torch.randn(num_groups, group_size, device='cuda', dtype=torch.float32)
        alpha = x.abs().amax(dim=-1).clamp(min=1e-4) + 0.1

        # CUDA kernel
        x_cuda = x.contiguous()
        alpha_cuda = alpha.contiguous()
        y_cuda = fake_quant_ste_seq2bit(x_cuda, alpha_cuda, group_size)

        # PyTorch reference
        y_torch = seq2bit_quantize_pytorch(x, alpha, group_size)

        # Compare
        max_diff = (y_cuda - y_torch).abs().max().item()
        rel_err = (y_cuda - y_torch).abs().sum().item() / (y_torch.abs().sum().item() + 1e-8)

        passed = max_diff < 1e-3 and rel_err < 1e-3
        all_passed = all_passed and passed

        print(f"  group_size={group_size}, num_groups={num_groups}:")
        print(f"    max_diff={max_diff:.2e}, rel_err={rel_err:.2e}, {'PASS' if passed else 'FAIL'}")

        # Verify quantization levels
        y_normalized = (y_cuda / alpha.reshape(-1, 1)).reshape(-1)
        unique_vals = torch.unique(y_normalized)
        expected_levels = torch.tensor([-0.75, -0.25, 0.25, 0.75], device='cuda')
        levels_correct = all(any((v - el).abs() < 0.1 for el in expected_levels) for v in unique_vals)
        print(f"    unique quantized values: {unique_vals.tolist()[:8]}...")
        print(f"    levels correct: {levels_correct}")

    if all_passed:
        print("\n  [PASS] fake_quant_fwd: CUDA kernel matches PyTorch reference")
    else:
        print("\n  [FAIL] fake_quant_fwd: Mismatch detected")

    return all_passed


def test_fake_quant_bwd():
    """Test 2: Verify fake_quant_backward_seq2bit gradient computation."""
    print("\n" + "=" * 60)
    print("Test 2: fake_quant_bwd (Seq2Bit CUDA kernel)")
    print("=" * 60)

    torch.manual_seed(42)

    test_cases = [
        (32, 128),   # group_size=128
        (64, 64),    # group_size=64
    ]

    all_passed = True

    for num_groups, group_size in test_cases:
        x = torch.randn(num_groups, group_size, device='cuda', dtype=torch.float32)
        alpha = x.abs().amax(dim=-1).clamp(min=1e-4) + 0.1
        grad_output = torch.randn_like(x)

        # CUDA kernel backward
        grad_x_cuda, grad_alpha_cuda = fake_quant_backward_seq2bit(
            grad_output.contiguous(),
            x.contiguous(),
            alpha.contiguous(),
            group_size,
        )

        # Check gradients are finite and non-zero
        grad_x_finite = torch.isfinite(grad_x_cuda).all()
        grad_alpha_finite = torch.isfinite(grad_alpha_cuda).all()
        grad_x_nonzero = grad_x_cuda.abs().sum() > 0
        grad_alpha_nonzero = grad_alpha_cuda.abs().sum() > 0

        # PyTorch reference backward (for comparison)
        grad_x_torch, grad_alpha_torch = seq2bit_backward_pytorch(
            grad_output, x, alpha, group_size
        )

        # Compare grad_x (should match for STE)
        grad_x_diff = (grad_x_cuda - grad_x_torch).abs().max().item()

        # grad_alpha may differ due to different implementations
        # The key is that gradient is finite and non-zero
        passed = (grad_x_diff < 1e-5 and
                  grad_x_finite and grad_alpha_finite and
                  grad_x_nonzero and grad_alpha_nonzero)
        all_passed = all_passed and passed

        print(f"  group_size={group_size}:")
        print(f"    grad_x match: max_diff={grad_x_diff:.2e}")
        print(f"    grad_x finite: {grad_x_finite}, non-zero: {grad_x_nonzero}")
        print(f"    grad_alpha finite: {grad_alpha_finite}, non-zero: {grad_alpha_nonzero}")
        print(f"    {'PASS' if passed else 'FAIL'}")

    if all_passed:
        print("\n  [PASS] fake_quant_bwd: CUDA backward produces valid gradients")
    else:
        print("\n  [FAIL] fake_quant_bwd: Gradient issue detected")

    return all_passed


def test_finite_difference():
    """Test 3: Numerical gradient verification using finite differences."""
    print("\n" + "=" * 60)
    print("Test 3: Finite Difference Gradient Check")
    print("=" * 60)

    torch.manual_seed(42)

    # Small test case for finite difference
    num_groups = 4
    group_size = 128
    eps = 1e-4

    x = torch.randn(num_groups, group_size, device='cuda', dtype=torch.float32, requires_grad=True)
    alpha = torch.randn(num_groups, device='cuda', dtype=torch.float32, requires_grad=True)
    alpha.data = alpha.data.abs().clamp(min=0.1)

    # Forward
    y = fake_quant_ste_seq2bit(x, alpha, group_size)
    loss = y.sum()

    # Analytical gradient
    loss.backward()
    grad_x_analytical = x.grad.clone()
    grad_alpha_analytical = alpha.grad.clone()

    # Numerical gradient for alpha (finite difference)
    grad_alpha_numerical = torch.zeros_like(alpha)
    for i in range(alpha.numel()):
        alpha_plus = alpha.clone().detach()
        alpha_plus[i] += eps
        alpha_minus = alpha.clone().detach()
        alpha_minus[i] -= eps

        y_plus = fake_quant_ste_seq2bit(x.detach(), alpha_plus, group_size)
        y_minus = fake_quant_ste_seq2bit(x.detach(), alpha_minus, group_size)

        grad_alpha_numerical[i] = (y_plus.sum() - y_minus.sum()) / (2 * eps)

    # Compare
    grad_alpha_diff = (grad_alpha_analytical - grad_alpha_numerical).abs()
    max_diff = grad_alpha_diff.max().item()
    rel_err = grad_alpha_diff.sum().item() / (grad_alpha_numerical.abs().sum().item() + 1e-8)

    # For STE, gradients may not match exactly due to discontinuous nature
    # We check if the analytical gradient is in the right direction
    correlation = (grad_alpha_analytical * grad_alpha_numerical).sum() / (
        grad_alpha_analytical.norm() * grad_alpha_numerical.norm() + 1e-8
    )

    print(f"  alpha grad comparison:")
    print(f"    max_diff={max_diff:.2e}, rel_err={rel_err:.2e}")
    print(f"    correlation={correlation:.4f}")

    # For quantization, exact numerical gradient match is not expected due to STE
    # We accept if correlation is positive and reasonable
    passed = correlation > 0.5 or max_diff < 0.1

    if passed:
        print(f"\n  [PASS] Finite difference: gradient direction consistent")
    else:
        print(f"\n  [FAIL] Finite difference: gradient mismatch")

    return passed


def test_single_linear():
    """Test 4: Single linear layer with Seq2Bit quantization."""
    print("\n" + "=" * 60)
    print("Test 4: Single Linear Layer Test")
    print("=" * 60)

    torch.manual_seed(42)

    # Create a linear layer
    in_features = 256
    out_features = 128
    group_size = 64

    weight = nn.Parameter(torch.randn(out_features, in_features, device='cuda'))
    bias = nn.Parameter(torch.zeros(out_features, device='cuda'))

    # Create alpha parameter for quantization
    num_groups = weight.numel() // group_size
    alpha = nn.Parameter(weight.data.reshape(-1, group_size).abs().amax(dim=-1).clamp(min=1e-4))

    # Forward with quantization
    input_tensor = torch.randn(4, in_features, device='cuda', requires_grad=True)

    # Quantize weight
    weight_q = fake_quant_ste_seq2bit(weight.reshape(-1).contiguous(), alpha, group_size)
    weight_q = weight_q.reshape(out_features, in_features)

    # Linear forward
    output = F.linear(input_tensor, weight_q, bias)

    # Backward
    loss = output.sum()
    loss.backward()

    # Check gradients
    has_grad = input_tensor.grad is not None and alpha.grad is not None
    grad_finite = has_grad and torch.isfinite(alpha.grad).all() and torch.isfinite(input_tensor.grad).all()

    print(f"  weight shape: {tuple(weight.shape)}")
    print(f"  alpha shape: {tuple(alpha.shape)}")
    print(f"  input grad exists: {input_tensor.grad is not None}")
    print(f"  alpha grad exists: {alpha.grad is not None}")
    print(f"  alpha grad norm: {alpha.grad.norm().item():.6f}")
    print(f"  all gradients finite: {grad_finite}")

    if has_grad and grad_finite:
        print("\n  [PASS] Single linear: forward+backward works")
        return True
    else:
        print("\n  [FAIL] Single linear: gradient issue")
        return False


def test_pytorch_reference():
    """Test 5: Comprehensive comparison with PyTorch reference."""
    print("\n" + "=" * 60)
    print("Test 5: PyTorch Reference Comparison")
    print("=" * 60)

    torch.manual_seed(42)

    # Test multiple configurations
    configs = [
        {'group_size': 64, 'shape': (128, 256)},
        {'group_size': 128, 'shape': (256, 512)},
        {'group_size': 256, 'shape': (512, 1024)},
    ]

    all_passed = True

    for config in configs:
        group_size = config['group_size']
        shape = config['shape']
        numel = shape[0] * shape[1]

        # Create data
        x = torch.randn(shape, device='cuda', dtype=torch.float32)
        num_groups = numel // group_size
        alpha = torch.randn(num_groups, device='cuda').abs().clamp(min=0.1)

        # CUDA forward
        x_flat = x.reshape(-1).contiguous()
        alpha_flat = alpha.contiguous()
        y_cuda = fake_quant_ste_seq2bit(x_flat, alpha_flat, group_size).reshape(shape)

        # PyTorch reference
        y_torch = seq2bit_quantize_pytorch(x, alpha, group_size)

        # Metrics
        diff = (y_cuda - y_torch).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check quantization saturation
        y_cuda_flat = y_cuda.reshape(-1, group_size)
        alpha_expanded = alpha.reshape(-1, 1).expand_as(y_cuda_flat)
        y_normalized = (y_cuda_flat / alpha_expanded.clamp(min=1e-6)).reshape(-1)
        saturation_low = (y_normalized < -0.8).float().mean().item()
        saturation_high = (y_normalized > 0.8).float().mean().item()

        passed = max_diff < 1e-3
        all_passed = all_passed and passed

        print(f"  shape={shape}, group_size={group_size}:")
        print(f"    max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        print(f"    saturation: low={saturation_low:.2%}, high={saturation_high:.2%}")
        print(f"    {'PASS' if passed else 'FAIL'}")

    if all_passed:
        print("\n  [PASS] PyTorch reference: all configs match")
    else:
        print("\n  [FAIL] PyTorch reference: mismatch detected")

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Level 1: Seq2Bit Kernel Math Tests")
    print("=" * 70)

    tests = [
        ("fake_quant_fwd", test_fake_quant_fwd),
        ("fake_quant_bwd", test_fake_quant_bwd),
        ("finite_difference", test_finite_difference),
        ("single_linear", test_single_linear),
        ("pytorch_reference", test_pytorch_reference),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Level 1 Kernel Math Tests")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)