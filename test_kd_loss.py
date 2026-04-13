#!/usr/bin/env python3
"""Test KD loss functions for Megatron QAT."""

import sys
import os

sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch

from megatron.core.quantization.megatron_kd import (
    mse_loss,
    fkld_loss,
    rkld_loss,
    gjsd_loss,
    get_kd_loss_func,
    MegatronKDLoss,
    HiddenStateCapture,
)


def test_mse_loss():
    """Test MSE loss function."""
    print("Test 1: MSE Loss")

    output = torch.randn(4, 512, device='cuda')
    target = torch.randn(4, 512, device='cuda')

    loss = mse_loss(output, target)

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"MSE loss should be non-negative, got {loss.item()}"

    print(f"  MSE loss: {loss.item():.6f}")
    print("  [PASS]")
    return True


def test_kl_losses():
    """Test FKLD and RKLD loss functions."""
    print("\nTest 2: KL Divergence Losses")

    output = torch.randn(4, 512, device='cuda')
    target = torch.randn(4, 512, device='cuda')

    fkld = fkld_loss(output, target)
    rkld = rkld_loss(output, target)

    print(f"  FKLD: {fkld.item():.6f}")
    print(f"  RKLD: {rkld.item():.6f}")
    print("  [PASS]")
    return True


def test_gjsd_loss():
    """Test GJSD loss function."""
    print("\nTest 3: GJSD Loss")

    output = torch.randn(4, 512, device='cuda')
    target = torch.randn(4, 512, device='cuda')

    loss = gjsd_loss(output, target, alpha=0.5)

    print(f"  GJSD (alpha=0.5): {loss.item():.6f}")

    # Test with different alpha
    loss_03 = gjsd_loss(output, target, alpha=0.3)
    print(f"  GJSD (alpha=0.3): {loss_03.item():.6f}")
    print("  [PASS]")
    return True


def test_get_kd_loss_func():
    """Test loss function factory."""
    print("\nTest 4: Loss Function Factory")

    loss_types = ['MSE', 'FKLD', 'RKLD', 'FKLD_RKLD', 'MSE_FKLD', 'MSE_RKLD', 'MSE_FKLD_RKLD', 'GJSD']

    output = torch.randn(4, 512, device='cuda')
    target = torch.randn(4, 512, device='cuda')

    for lt in loss_types:
        func = get_kd_loss_func(lt)
        loss = func(output, target)
        print(f"  {lt}: {loss.item():.6f}")

    print("  [PASS]")
    return True


def test_megatron_kd_loss_module():
    """Test MegatronKDLoss module."""
    print("\nTest 5: MegatronKDLoss Module")

    kd_loss = MegatronKDLoss(
        loss_type='MSE',
        kd_layers=[0, 1, 2, 3],
        kd_weight=0.5,
    )

    # Create fake hidden states
    student_hiddens = {
        0: torch.randn(2, 128, 512, device='cuda'),
        1: torch.randn(2, 128, 512, device='cuda'),
        2: torch.randn(2, 128, 512, device='cuda'),
        3: torch.randn(2, 128, 512, device='cuda'),
    }
    teacher_hiddens = {
        0: torch.randn(2, 128, 512, device='cuda'),
        1: torch.randn(2, 128, 512, device='cuda'),
        2: torch.randn(2, 128, 512, device='cuda'),
        3: torch.randn(2, 128, 512, device='cuda'),
    }

    total_loss, per_layer_loss = kd_loss(student_hiddens, teacher_hiddens)

    print(f"  Total loss (weighted): {total_loss.item():.6f}")
    print(f"  Per-layer losses: {len(per_layer_loss)} layers")
    for idx, loss in per_layer_loss.items():
        print(f"    Layer {idx}: {loss.item():.6f}")

    assert total_loss.ndim == 0, "Total loss should be scalar"
    assert len(per_layer_loss) == 4, "Should have 4 layer losses"
    print("  [PASS]")
    return True


def test_hidden_state_capture():
    """Test HiddenStateCapture context manager."""
    print("\nTest 6: HiddenStateCapture")

    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 64) for _ in range(4)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = SimpleModel().cuda()

    capture = HiddenStateCapture(model, layer_attr='layers')

    with capture:
        output = model(torch.randn(2, 64, device='cuda'))

    print(f"  Captured {len(capture.hidden_states)} layer outputs")
    for idx, h in capture.hidden_states.items():
        print(f"    Layer {idx}: shape {tuple(h.shape)}")

    assert len(capture.hidden_states) == 4, "Should capture 4 layers"
    print("  [PASS]")
    return True


def main():
    print("=" * 60)
    print("Megatron KD Loss Tests")
    print("=" * 60)

    tests = [
        test_mse_loss,
        test_kl_losses,
        test_gjsd_loss,
        test_get_kd_loss_func,
        test_megatron_kd_loss_module,
        test_hidden_state_capture,
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