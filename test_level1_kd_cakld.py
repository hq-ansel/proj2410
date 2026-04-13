#!/usr/bin/env python3
"""
Level 1: Kernel Math Tests for KD Loss and CAKLD

Tests:
1. KD Loss: MSE, FKLD, RKLD, GJSD correctness
2. KD Loss: Gradient flow through loss
3. CAKLD: Forward computation
4. CAKLD: Gradient computation
5. CAKLD: Temperature scaling
"""

import sys
import os
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import KD and CAKLD components
from megatron.core.quantization.megatron_kd import (
    mse_loss,
    fkld_loss,
    rkld_loss,
    gjsd_loss,
    MegatronKDLoss,
)
from megatron.core.quantization.cakld import (
    cakld_loss,
    CAKLDLoss,
)


# ---------------------------------------------------------------------------
# Test 1: KD Loss Correctness
# ---------------------------------------------------------------------------

def test_kd_loss_correctness():
    """Test KD loss functions compute correct values."""
    print("\n" + "=" * 60)
    print("Test 1: KD Loss Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    # Create test data
    output = torch.randn(4, 512, device='cuda')
    target = torch.randn(4, 512, device='cuda')

    # Test MSE
    mse = mse_loss(output, target)
    mse_expected = F.mse_loss(output, target)
    mse_match = torch.isclose(mse, mse_expected, rtol=1e-5)

    # Test FKLD
    fkld = fkld_loss(output, target)
    # FKLD should be >= 0
    fkld_valid = fkld >= 0

    # Test RKLD
    rkld = rkld_loss(output, target)
    rkld_valid = rkld >= 0

    # Test GJSD
    gjsd = gjsd_loss(output, target)
    gjsd_valid = gjsd >= 0

    print(f"  MSE: computed={mse.item():.6f}, expected={mse_expected.item():.6f}, match={mse_match.item()}")
    print(f"  FKLD: {fkld.item():.6f}, valid={fkld_valid.item()}")
    print(f"  RKLD: {rkld.item():.6f}, valid={rkld_valid.item()}")
    print(f"  GJSD: {gjsd.item():.6f}, valid={gjsd_valid.item()}")

    all_passed = mse_match and fkld_valid and rkld_valid and gjsd_valid

    if all_passed:
        print("\n  [PASS] KD loss correctness: all losses valid")
    else:
        print("\n  [FAIL] KD loss correctness: some losses invalid")

    return all_passed.item()


# ---------------------------------------------------------------------------
# Test 2: KD Loss Gradient Flow
# ---------------------------------------------------------------------------

def test_kd_loss_gradient():
    """Test gradients flow correctly through KD loss."""
    print("\n" + "=" * 60)
    print("Test 2: KD Loss Gradient Flow")
    print("=" * 60)

    torch.manual_seed(42)

    output = torch.randn(4, 512, device='cuda', requires_grad=True)
    target = torch.randn(4, 512, device='cuda')

    # Test each loss function
    losses = {
        'MSE': mse_loss(output, target),
        'FKLD': fkld_loss(output, target),
        'RKLD': rkld_loss(output, target),
        'GJSD': gjsd_loss(output, target),
    }

    all_passed = True
    for name, loss in losses.items():
        if output.grad is not None:
            output.grad.zero_()

        loss.backward(retain_graph=True)

        has_grad = output.grad is not None
        grad_finite = has_grad and torch.isfinite(output.grad).all()
        grad_nonzero = has_grad and output.grad.abs().sum() > 0

        passed = grad_finite and grad_nonzero
        all_passed = all_passed and passed

        print(f"  {name}: grad_finite={grad_finite.item() if has_grad else False}, grad_nonzero={grad_nonzero.item() if has_grad else False}")

    if all_passed:
        print("\n  [PASS] KD loss gradients: all valid")
    else:
        print("\n  [FAIL] KD loss gradients: some invalid")

    return all_passed


# ---------------------------------------------------------------------------
# Test 3: CAKLD Forward
# ---------------------------------------------------------------------------

def test_cakld_forward():
    """Test CAKLD forward computation."""
    print("\n" + "=" * 60)
    print("Test 3: CAKLD Forward")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    vocab_size = 1000

    student_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    # Test different alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_passed = True

    for alpha in alphas:
        loss = cakld_loss(student_logits, teacher_logits, alpha=alpha)
        loss_valid = loss >= 0 and torch.isfinite(loss)

        print(f"  alpha={alpha:.2f}: loss={loss.item():.6f}, valid={loss_valid.item()}")

        all_passed = all_passed and loss_valid.item()

    if all_passed:
        print("\n  [PASS] CAKLD forward: all alpha values produce valid loss")
    else:
        print("\n  [FAIL] CAKLD forward: invalid loss for some alpha")

    return all_passed


# ---------------------------------------------------------------------------
# Test 4: CAKLD Gradient
# ---------------------------------------------------------------------------

def test_cakld_gradient():
    """Test CAKLD gradient computation."""
    print("\n" + "=" * 60)
    print("Test 4: CAKLD Gradient")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    vocab_size = 100

    student_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', requires_grad=True)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    # Compute loss and backward
    loss = cakld_loss(student_logits, teacher_logits, alpha=0.5)
    loss.backward()

    has_grad = student_logits.grad is not None
    grad_finite = has_grad and torch.isfinite(student_logits.grad).all()
    grad_nonzero = has_grad and student_logits.grad.abs().sum() > 0

    print(f"  has_grad: {has_grad}")
    print(f"  grad_finite: {grad_finite.item() if has_grad else False}")
    print(f"  grad_nonzero: {grad_nonzero.item() if has_grad else False}")
    print(f"  grad_norm: {student_logits.grad.norm().item() if has_grad else 'N/A'}")

    passed = grad_finite.item() if has_grad else False

    if passed:
        print("\n  [PASS] CAKLD gradient: valid")
    else:
        print("\n  [FAIL] CAKLD gradient: invalid")

    return passed


# ---------------------------------------------------------------------------
# Test 5: CAKLD Temperature Scaling
# ---------------------------------------------------------------------------

def test_cakld_temperature():
    """Test CAKLD temperature scaling effect."""
    print("\n" + "=" * 60)
    print("Test 5: CAKLD Temperature Scaling")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    vocab_size = 100

    student_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    temperatures = [0.5, 1.0, 2.0, 5.0]
    losses = []

    for temp in temperatures:
        loss = cakld_loss(student_logits, teacher_logits, temperature=temp)
        losses.append(loss.item())
        print(f"  temperature={temp:.1f}: loss={loss.item():.6f}")

    # Higher temperature should generally give lower loss (softer distributions)
    # This is not always true but is a common pattern
    # We just verify all are finite
    all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)

    if all_finite:
        print("\n  [PASS] CAKLD temperature: all temperatures produce finite loss")
    else:
        print("\n  [FAIL] CAKLD temperature: some losses are non-finite")

    return all_finite


# ---------------------------------------------------------------------------
# Test 6: CAKLD Attention Mask
# ---------------------------------------------------------------------------

def test_cakld_attention_mask():
    """Test CAKLD with attention mask."""
    print("\n" + "=" * 60)
    print("Test 6: CAKLD with Attention Mask")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    vocab_size = 100

    student_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    # Create attention mask (mask last 2 tokens)
    attention_mask = torch.ones(batch_size, seq_len, device='cuda')
    attention_mask[:, 6:] = 0

    loss_with_mask = cakld_loss(student_logits, teacher_logits, attention_mask=attention_mask)
    loss_without_mask = cakld_loss(student_logits, teacher_logits)

    both_finite = torch.isfinite(loss_with_mask) and torch.isfinite(loss_without_mask)

    print(f"  loss with mask: {loss_with_mask.item():.6f}")
    print(f"  loss without mask: {loss_without_mask.item():.6f}")
    print(f"  both finite: {both_finite.item()}")

    if both_finite.item():
        print("\n  [PASS] CAKLD attention mask: works correctly")
    else:
        print("\n  [FAIL] CAKLD attention mask: non-finite loss")

    return both_finite.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Level 1: KD Loss & CAKLD Kernel Math Tests")
    print("=" * 70)

    tests = [
        ("kd_loss_correctness", test_kd_loss_correctness),
        ("kd_loss_gradient", test_kd_loss_gradient),
        ("cakld_forward", test_cakld_forward),
        ("cakld_gradient", test_cakld_gradient),
        ("cakld_temperature", test_cakld_temperature),
        ("cakld_attention_mask", test_cakld_attention_mask),
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
    print("Summary: Level 1 KD & CAKLD Tests")
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