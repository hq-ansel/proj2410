#!/usr/bin/env python3
"""Test CAKLD loss functions for Megatron QAT."""

import sys
import os

sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch

from megatron.core.quantization.cakld import (
    cakld_loss,
    attention_weighted_cakld,
    CAKLDLoss,
    MultiLayerCAKLD,
    CalibrationDataProcessor,
)


def test_cakld_loss():
    """Test basic CAKLD loss."""
    print("Test 1: Basic CAKLD Loss")

    # Simulate logits [batch=2, seq_len=8, vocab_size=100]
    student_logits = torch.randn(2, 8, 100, device='cuda')
    teacher_logits = torch.randn(2, 8, 100, device='cuda')

    loss = cakld_loss(student_logits, teacher_logits)

    print(f"  CAKLD loss: {loss.item():.6f}")
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative"
    print("  [PASS]")
    return True


def test_cakld_with_attention_mask():
    """Test CAKLD with attention mask."""
    print("\nTest 2: CAKLD with Attention Mask")

    student_logits = torch.randn(2, 8, 100, device='cuda')
    teacher_logits = torch.randn(2, 8, 100, device='cuda')
    attention_mask = torch.ones(2, 8, device='cuda')
    attention_mask[:, 6:] = 0  # Mask last 2 tokens

    loss_with_mask = cakld_loss(student_logits, teacher_logits, attention_mask=attention_mask)
    loss_without_mask = cakld_loss(student_logits, teacher_logits)

    print(f"  Loss with mask: {loss_with_mask.item():.6f}")
    print(f"  Loss without mask: {loss_without_mask.item():.6f}")
    print("  [PASS]")
    return True


def test_cakld_alpha():
    """Test CAKLD with different alpha values."""
    print("\nTest 3: CAKLD Alpha Parameter")

    student_logits = torch.randn(2, 8, 100, device='cuda')
    teacher_logits = torch.randn(2, 8, 100, device='cuda')

    # alpha=0.5 is symmetric (like JS divergence)
    loss_05 = cakld_loss(student_logits, teacher_logits, alpha=0.5)

    # alpha=1.0 is pure forward KL
    loss_10 = cakld_loss(student_logits, teacher_logits, alpha=1.0)

    # alpha=0.0 is pure reverse KL
    loss_00 = cakld_loss(student_logits, teacher_logits, alpha=0.0)

    print(f"  alpha=0.5 (symmetric): {loss_05.item():.6f}")
    print(f"  alpha=1.0 (forward KL): {loss_10.item():.6f}")
    print(f"  alpha=0.0 (reverse KL): {loss_00.item():.6f}")
    print("  [PASS]")
    return True


def test_temperature():
    """Test CAKLD with temperature scaling."""
    print("\nTest 4: Temperature Scaling")

    student_logits = torch.randn(2, 8, 100, device='cuda')
    teacher_logits = torch.randn(2, 8, 100, device='cuda')

    loss_t1 = cakld_loss(student_logits, teacher_logits, temperature=1.0)
    loss_t2 = cakld_loss(student_logits, teacher_logits, temperature=2.0)
    loss_t05 = cakld_loss(student_logits, teacher_logits, temperature=0.5)

    print(f"  temperature=1.0: {loss_t1.item():.6f}")
    print(f"  temperature=2.0: {loss_t2.item():.6f}")
    print(f"  temperature=0.5: {loss_t05.item():.6f}")
    print("  [PASS]")
    return True


def test_cakld_loss_module():
    """Test CAKLDLoss module."""
    print("\nTest 5: CAKLDLoss Module")

    cakld = CAKLDLoss(
        temperature=1.0,
        alpha=0.5,
        use_attention_weighting=False,
    )

    student_logits = torch.randn(2, 8, 100, device='cuda')
    teacher_logits = torch.randn(2, 8, 100, device='cuda')

    loss = cakld(student_logits, teacher_logits)

    print(f"  Module loss: {loss.item():.6f}")
    print("  [PASS]")
    return True


def test_multi_layer_cakld():
    """Test MultiLayerCAKLD."""
    print("\nTest 6: MultiLayerCAKLD")

    cakld = MultiLayerCAKLD(
        temperature=1.0,
        alpha=0.5,
        calibrate_layers=[0, 1, 2],
        layer_weight_strategy='uniform',
    )

    student_hiddens = {
        0: torch.randn(2, 8, 64, device='cuda'),
        1: torch.randn(2, 8, 64, device='cuda'),
        2: torch.randn(2, 8, 64, device='cuda'),
    }
    teacher_hiddens = {
        0: torch.randn(2, 8, 64, device='cuda'),
        1: torch.randn(2, 8, 64, device='cuda'),
        2: torch.randn(2, 8, 64, device='cuda'),
    }

    total_loss, per_layer = cakld(student_hiddens, teacher_hiddens)

    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  Per-layer losses: {per_layer}")
    print("  [PASS]")
    return True


def test_calibration_processor():
    """Test CalibrationDataProcessor."""
    print("\nTest 7: CalibrationDataProcessor")

    processor = CalibrationDataProcessor(
        max_calib_tokens=4096,
        importance_metric='entropy',
    )

    logits = torch.randn(2, 8, 100, device='cuda')
    attention_mask = torch.ones(2, 8, device='cuda')

    weights = processor.compute_importance_weights(logits, attention_mask)

    print(f"  Importance weights shape: {tuple(weights.shape)}")
    print(f"  Weights sum: {weights.sum().item():.6f}")
    print("  [PASS]")
    return True


def main():
    print("=" * 60)
    print("CAKLD Loss Tests")
    print("=" * 60)

    tests = [
        test_cakld_loss,
        test_cakld_with_attention_mask,
        test_cakld_alpha,
        test_temperature,
        test_cakld_loss_module,
        test_multi_layer_cakld,
        test_calibration_processor,
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