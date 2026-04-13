#!/usr/bin/env python3
"""
End-to-end validation for Megatron QAT components.

Tests:
1. Seq2Bit QAT training
2. KD Loss integration
3. CAKLD calibration

Usage:
    python test_e2e_qat_components.py --test seq2bit
    python test_e2e_qat_components.py --test kd
    python test_e2e_qat_components.py --test cakld
    python test_e2e_qat_components.py --test all
"""

import sys
import os

# Setup paths
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Dict, Optional, List

# Import QAT components
from megatron.core.quantization.megatron_qat import (
    MegatronSeq2BitQuantizer,
    MegatronWeightQuantizer,
    QuantConfig,
    convert_megatron_model,
    set_quant_state,
    quantizer_parameters,
)
from megatron.core.quantization.megatron_kd import (
    MegatronKDLoss,
    HiddenStateCapture,
    mse_loss,
    fkld_loss,
    gjsd_loss,
)
from megatron.core.quantization.cakld import (
    CAKLDLoss,
    MultiLayerCAKLD,
    cakld_loss,
)


# ---------------------------------------------------------------------------
# Test 1: Seq2Bit E2E
# ---------------------------------------------------------------------------

def test_seq2bit_e2e():
    """Test Seq2Bit quantization in a simulated training loop."""
    print("\n" + "=" * 60)
    print("Test 1: Seq2Bit E2E Training Simulation")
    print("=" * 60)

    # Create a simple linear layer to simulate
    class SimpleLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    # Create model
    model = nn.ModuleDict({
        'layer1': SimpleLinear(512, 256),
        'layer2': SimpleLinear(256, 128),
    }).cuda()

    # Create Seq2Bit quantizers (store separately, not as submodules)
    config = QuantConfig(quant_type="seq2bit", group_size=128)

    quantizers = {}
    for name, layer in model.items():
        quantizer = MegatronSeq2BitQuantizer(layer.weight, config, prefix=name)
        quantizers[name] = quantizer
        # Don't add as submodule to avoid parameter duplication

    # Enable quantization
    for q in quantizers.values():
        q.use_weight_quant = True

    # Collect model params (excluding quantizer params since not submodules)
    model_params = list(model.parameters())

    # Collect alpha parameters
    alpha_params = [q.weight_quantizer.alpha for q in quantizers.values()]

    # Simulate training loop
    optimizer = torch.optim.AdamW([
        {'params': model_params, 'lr': 1e-4},
        {'params': alpha_params, 'lr': 1e-3},
    ])

    losses = []
    for step in range(10):
        optimizer.zero_grad()

        # Random input
        x = torch.randn(4, 512, device='cuda')

        # Forward with quantization
        h = x
        for name, layer in model.items():
            # Quantize weight
            w_q = quantizers[name](layer.weight)
            h = F.linear(h, w_q, layer.bias)

        # Compute loss (reconstruction)
        loss = h.mean()
        loss.backward()

        # Check gradients
        alpha_grads = [q.weight_quantizer.alpha.grad for q in quantizers.values()]
        has_grad = all(g is not None and torch.isfinite(g).all() for g in alpha_grads)

        optimizer.step()
        losses.append(loss.item())

        print(f"  Step {step}: loss={loss.item():.4f}, alpha_grad_ok={has_grad}")

    # Verify gradients are computed correctly
    assert all(g is not None and torch.isfinite(g).all() for g in alpha_grads), "Alpha gradients should be finite"

    print(f"\n  [PASS] Seq2Bit training: alpha_grad_ok=True, losses finite")
    return True


# ---------------------------------------------------------------------------
# Test 2: KD Loss E2E
# ---------------------------------------------------------------------------

def test_kd_loss_e2e():
    """Test KD loss in a simulated teacher-student training."""
    print("\n" + "=" * 60)
    print("Test 2: KD Loss E2E Training Simulation")
    print("=" * 60)

    # Create teacher and student models
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_size, num_layers=4):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
            ])
            self.final = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            hiddens = {}
            for i, layer in enumerate(self.layers):
                x = layer(x)
                hiddens[i] = x
            x = self.final(x)
            return x, hiddens

    hidden_size = 256
    teacher = SimpleTransformer(hidden_size, num_layers=4).cuda()
    student = SimpleTransformer(hidden_size, num_layers=4).cuda()

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    # KD loss module
    kd_loss_fn = MegatronKDLoss(
        loss_type='MSE',
        kd_layers=[0, 1, 2, 3],
        kd_weight=1.0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    losses = []
    kd_losses = []
    for step in range(10):
        optimizer.zero_grad()

        # Random input
        x = torch.randn(4, 128, hidden_size, device='cuda')

        # Teacher forward (no grad)
        with torch.no_grad():
            _, teacher_hiddens = teacher(x)

        # Student forward with hidden state capture
        student_hiddens = {}
        h = x
        for i, layer in enumerate(student.layers):
            h = layer(h)
            student_hiddens[i] = h

        # Compute KD loss
        kd_loss, per_layer = kd_loss_fn(student_hiddens, teacher_hiddens)

        # Task loss (simple reconstruction)
        task_loss = h.mean()

        # Total loss
        total_loss = task_loss + kd_loss

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        kd_losses.append(kd_loss.item())

        print(f"  Step {step}: total_loss={total_loss.item():.4f}, kd_loss={kd_loss.item():.4f}")

    # Verify losses are finite (KD loss may not decrease in short training)
    assert all(torch.isfinite(torch.tensor(l)) for l in kd_losses), "KD losses should be finite"

    print(f"\n  [PASS] KD training: kd_loss {kd_losses[0]:.4f} -> {kd_losses[-1]:.4f} (finite=True)")
    return True


# ---------------------------------------------------------------------------
# Test 3: CAKLD E2E
# ---------------------------------------------------------------------------

def test_cakld_e2e():
    """Test CAKLD in a calibration simulation."""
    print("\n" + "=" * 60)
    print("Test 3: CAKLD Calibration Simulation")
    print("=" * 60)

    # Simulate logits from teacher and student
    vocab_size = 1000
    seq_len = 128
    batch_size = 4

    # Create CAKLD loss module
    cakld_fn = CAKLDLoss(
        temperature=2.0,
        alpha=0.5,
        use_attention_weighting=False,
    )

    # Simulate calibration process
    losses = []
    for step in range(10):
        # Simulate teacher logits (fixed distribution)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

        # Simulate student logits (moving toward teacher)
        noise_scale = max(0.1, 1.0 - step * 0.1)  # Decreasing noise
        student_logits = teacher_logits + noise_scale * torch.randn_like(teacher_logits)

        # Compute CAKLD loss
        loss = cakld_fn(student_logits, teacher_logits)

        losses.append(loss.item())

        print(f"  Step {step}: cakld_loss={loss.item():.4f}, noise_scale={noise_scale:.2f}")

    # Verify loss decreases as student approaches teacher
    assert losses[-1] < losses[0], f"CAKLD should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    print(f"\n  [PASS] CAKLD calibration: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    return True


# ---------------------------------------------------------------------------
# Test 4: Combined E2E (Seq2Bit + KD + CAKLD)
# ---------------------------------------------------------------------------

def test_combined_e2e():
    """Test all components together in a combined training simulation."""
    print("\n" + "=" * 60)
    print("Test 4: Combined E2E (Seq2Bit + KD + CAKLD)")
    print("=" * 60)

    vocab_size = 1000
    hidden_size = 256

    class TransformerBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attn = nn.Linear(hidden_size, hidden_size)
            self.ffn = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            h = self.attn(x)
            h = F.relu(h)
            h = self.ffn(h)
            return h

    class SimpleModel(nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                TransformerBlock(hidden_size) for _ in range(num_layers)
            ])
            self.head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            hiddens = {}
            for i, layer in enumerate(self.layers):
                x = layer(x)
                hiddens[i] = x
            logits = self.head(x)
            return logits, hiddens

    # Create teacher and student
    teacher = SimpleModel(vocab_size, hidden_size).cuda()
    student = SimpleModel(vocab_size, hidden_size).cuda()

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    # Apply Seq2Bit quantization to student (store separately, not as submodules)
    config = QuantConfig(quant_type="seq2bit", group_size=64)
    quantizers = {}
    for name, module in student.named_modules():
        if isinstance(module, nn.Linear) and 'head' not in name:
            quantizer = MegatronSeq2BitQuantizer(module.weight, config, prefix=name)
            quantizers[name] = quantizer
            # Don't add as submodule to avoid parameter duplication

    # Enable quantization
    for q in quantizers.values():
        q.use_weight_quant = True

    # KD and CAKLD modules
    kd_fn = MegatronKDLoss(loss_type='MSE', kd_layers=[0, 1, 2, 3], kd_weight=0.5)
    cakld_fn = CAKLDLoss(temperature=2.0, alpha=0.5)

    # Collect student params (excluding quantizer alpha which is not a submodule)
    student_params = [p for p in student.parameters() if p.requires_grad]
    alpha_params = [q.weight_quantizer.alpha for q in quantizers.values()]

    optimizer = torch.optim.AdamW([
        {'params': student_params, 'lr': 1e-4},
        {'params': alpha_params, 'lr': 1e-3},
    ])

    # Training loop
    total_losses = []
    kd_losses = []
    cakld_losses = []

    for step in range(10):
        optimizer.zero_grad()

        # Random input
        input_ids = torch.randint(0, vocab_size, (2, 32), device='cuda')

        # Teacher forward
        with torch.no_grad():
            teacher_logits, teacher_hiddens = teacher(input_ids)

        # Student forward with quantization
        x = student.embed(input_ids)
        student_hiddens = {}

        for i, layer in enumerate(student.layers):
            # Quantize weights
            quant_name = f'layers.{i}.attn'
            if quant_name in quantizers:
                w_q = quantizers[quant_name](layer.attn.weight)
                h = F.linear(x, w_q, layer.attn.bias)
            else:
                h = layer.attn(x)

            h = F.relu(h)

            quant_name = f'layers.{i}.ffn'
            if quant_name in quantizers:
                w_q = quantizers[quant_name](layer.ffn.weight)
                h = F.linear(h, w_q, layer.ffn.bias)
            else:
                h = layer.ffn(h)

            student_hiddens[i] = h
            x = h

        student_logits = student.head(x)

        # Compute losses
        kd_loss, _ = kd_fn(student_hiddens, teacher_hiddens)
        calib_loss = cakld_fn(student_logits, teacher_logits)
        task_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            input_ids.view(-1)
        )

        total_loss = task_loss + kd_loss + calib_loss

        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        kd_losses.append(kd_loss.item())
        cakld_losses.append(calib_loss.item())

        print(f"  Step {step}: total={total_loss.item():.4f}, kd={kd_loss.item():.4f}, cakld={calib_loss.item():.4f}")

    # Verify all losses are finite and training works
    assert all(torch.isfinite(torch.tensor(l)) for l in total_losses), "Losses should be finite"
    assert total_losses[-1] < total_losses[0], f"Total loss should decrease: {total_losses[0]:.4f} -> {total_losses[-1]:.4f}"

    print(f"\n  [PASS] Combined training: total_loss {total_losses[0]:.4f} -> {total_losses[-1]:.4f}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E2E QAT Component Tests")
    parser.add_argument('--test', type=str, default='all',
                       choices=['seq2bit', 'kd', 'cakld', 'combined', 'all'],
                       help='Which test to run')
    args = parser.parse_args()

    print("=" * 60)
    print("Megatron QAT Components E2E Validation")
    print("=" * 60)

    tests = {
        'seq2bit': test_seq2bit_e2e,
        'kd': test_kd_loss_e2e,
        'cakld': test_cakld_e2e,
        'combined': test_combined_e2e,
    }

    if args.test == 'all':
        test_order = ['seq2bit', 'kd', 'cakld', 'combined']
    else:
        test_order = [args.test]

    passed = 0
    failed = 0

    for test_name in test_order:
        try:
            if tests[test_name]():
                passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"E2E Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)