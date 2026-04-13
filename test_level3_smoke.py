#!/usr/bin/env python3
"""
Level 3: Smoke Test - Tiny GPT Model with QAT

Tests:
1. 2-layer tiny GPT with Seq2Bit quantization
2. Loss decreases during training
3. Memory stability (no leaks)
4. All QAT components work together
"""

import sys
import os
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410")
sys.path.insert(0, "/home/ubuntu/data/exp/proj2410/Megatron-LM")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import gc


# ---------------------------------------------------------------------------
# Tiny GPT Model
# ---------------------------------------------------------------------------

class TinyGPTBlock(nn.Module):
    """Single transformer block with QAT quantization support."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # FFN
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size, bias=False)

        # Norm
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Quantizers (set externally)
        self._quantizers = {}

    def set_quantizers(self, quantizers):
        self._quantizers = quantizers

    def _quantized_linear(self, name, x, weight, bias=None):
        """Apply linear with quantized weight."""
        if name in self._quantizers:
            w_q = self._quantizers[name](weight)
        else:
            w_q = weight
        return F.linear(x, w_q, bias)

    def forward(self, x):
        # Attention
        residual = x
        x = self.ln1(x)

        B, S, D = x.shape

        # Apply quantization to weights, not outputs
        q = self._quantized_linear('q_proj', x, self.q_proj.weight)
        k = self._quantized_linear('k_proj', x, self.k_proj.weight)
        v = self._quantized_linear('v_proj', x, self.v_proj.weight)

        # Reshape for attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self._quantized_linear('o_proj', out, self.o_proj.weight)
        x = residual + out

        # FFN
        residual = x
        x = self.ln2(x)

        gate = F.silu(self._quantized_linear('gate_proj', x, self.gate_proj.weight))
        up = self._quantized_linear('up_proj', x, self.up_proj.weight)

        ffn_out = gate * up
        x = residual + self._quantized_linear('down_proj', ffn_out, self.down_proj.weight)

        return x


class TinyGPT(nn.Module):
    """Tiny GPT model for smoke testing."""

    def __init__(self, vocab_size, hidden_size, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TinyGPTBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------

def get_memory_mb():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def test_tiny_gpt_training():
    """Test 1: Tiny GPT with Seq2Bit training."""
    print("\n" + "=" * 60)
    print("Test 1: Tiny GPT Training with Seq2Bit")
    print("=" * 60)

    from megatron.core.quantization.megatron_qat import QuantConfig, MegatronSeq2BitQuantizer

    torch.manual_seed(42)

    # Model config
    vocab_size = 1000
    hidden_size = 128
    num_heads = 4
    num_layers = 2

    # Create model
    model = TinyGPT(vocab_size, hidden_size, num_heads, num_layers).cuda()
    print(f"  Model: vocab={vocab_size}, hidden={hidden_size}, heads={num_heads}, layers={num_layers}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Create quantizers for all linear layers
    config = QuantConfig(quant_type="seq2bit", group_size=64)

    # Organize quantizers by layer
    layer_quantizers = [{} for _ in range(num_layers)]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            q = MegatronSeq2BitQuantizer(module.weight, config, prefix=name)
            q.use_weight_quant = True

            # Determine which layer this belongs to
            if 'layers.' in name:
                layer_idx = int(name.split('.')[1])
                proj_name = name.split('.')[-1]
                layer_quantizers[layer_idx][proj_name] = q

    # Set quantizers to each layer
    for i, layer in enumerate(model.layers):
        layer.set_quantizers(layer_quantizers[i])

    # Collect all alpha params
    all_alpha_params = []
    for lq in layer_quantizers:
        for q in lq.values():
            all_alpha_params.append(q.weight_quantizer.alpha)

    print(f"  Quantized layers: {sum(len(lq) for lq in layer_quantizers)}")

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': all_alpha_params, 'lr': 1e-3},
    ])

    # Training loop
    batch_size = 4
    seq_len = 32
    num_steps = 20

    losses = []
    memories = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Random input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        labels = input_ids.clone()

        # Forward
        logits = model(input_ids)

        # Loss
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        # Backward
        loss.backward()

        # Check gradients
        alpha_grads_ok = all(
            q.weight_quantizer.alpha.grad is not None and
            torch.isfinite(q.weight_quantizer.alpha.grad).all()
            for lq in layer_quantizers for q in lq.values()
        )

        optimizer.step()

        # Record
        losses.append(loss.item())
        memories.append(get_memory_mb())

        if step % 5 == 0:
            print(f"  Step {step:2d}: loss={loss.item():.4f}, mem={memories[-1]:.1f}MB, alpha_grad_ok={alpha_grads_ok}")

    # Verify loss decreases
    loss_decreased = losses[-1] < losses[0]
    print(f"\n  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (decreased={loss_decreased})")

    # Verify memory stable
    mem_growth = memories[-1] - memories[0]
    mem_stable = mem_growth < 100  # Less than 100MB growth
    print(f"  Memory: {memories[0]:.1f}MB -> {memories[-1]:.1f}MB (growth={mem_growth:.1f}MB, stable={mem_stable})")

    passed = loss_decreased and mem_stable and alpha_grads_ok

    if passed:
        print("\n  [PASS] Tiny GPT training: loss decreases, memory stable")
    else:
        print("\n  [FAIL] Tiny GPT training: issues detected")

    return passed


def test_kd_integration():
    """Test 2: KD Loss integration with tiny model."""
    print("\n" + "=" * 60)
    print("Test 2: KD Loss Integration")
    print("=" * 60)

    from megatron.core.quantization.megatron_qat import QuantConfig, MegatronSeq2BitQuantizer
    from megatron.core.quantization.megatron_kd import MegatronKDLoss

    torch.manual_seed(42)

    # Create teacher and student
    vocab_size = 500
    hidden_size = 64
    num_heads = 2
    num_layers = 2

    teacher = TinyGPT(vocab_size, hidden_size, num_heads, num_layers).cuda()
    student = TinyGPT(vocab_size, hidden_size, num_heads, num_layers).cuda()

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    # Quantize student
    config = QuantConfig(quant_type="seq2bit", group_size=64)
    layer_quantizers = [{} for _ in range(num_layers)]

    for name, module in student.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            q = MegatronSeq2BitQuantizer(module.weight, config, prefix=name)
            q.use_weight_quant = True
            if 'layers.' in name:
                layer_idx = int(name.split('.')[1])
                proj_name = name.split('.')[-1]
                layer_quantizers[layer_idx][proj_name] = q

    for i, layer in enumerate(student.layers):
        layer.set_quantizers(layer_quantizers[i])

    # Collect alpha params
    all_alpha_params = []
    for lq in layer_quantizers:
        for q in lq.values():
            all_alpha_params.append(q.weight_quantizer.alpha)

    # KD loss
    kd_loss_fn = MegatronKDLoss(loss_type='MSE', kd_weight=0.5)

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': student.parameters(), 'lr': 1e-4},
        {'params': all_alpha_params, 'lr': 1e-3},
    ])

    # Training
    losses = []
    kd_losses = []

    for step in range(10):
        optimizer.zero_grad()

        input_ids = torch.randint(0, vocab_size, (2, 16), device='cuda')

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = teacher(input_ids)

        # Student forward
        student_logits = student(input_ids)

        # Losses
        lm_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            input_ids.view(-1)
        )

        # KD loss (simplified - using random hidden states for testing)
        batch_size, seq_len = input_ids.shape
        student_hiddens = {0: torch.randn(batch_size, seq_len, hidden_size, device='cuda')}
        teacher_hiddens = {0: torch.randn(batch_size, seq_len, hidden_size, device='cuda')}

        kd_loss, _ = kd_loss_fn(student_hiddens, teacher_hiddens)

        total_loss = lm_loss + kd_loss

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        kd_losses.append(kd_loss.item())

        if step % 5 == 0:
            print(f"  Step {step}: total_loss={total_loss.item():.4f}, kd_loss={kd_loss.item():.4f}")

    # Verify
    all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)

    if all_finite:
        print(f"\n  [PASS] KD integration: all losses finite")
        return True
    else:
        print(f"\n  [FAIL] KD integration: non-finite losses")
        return False


def test_cakld_integration():
    """Test 3: CAKLD integration with tiny model."""
    print("\n" + "=" * 60)
    print("Test 3: CAKLD Integration")
    print("=" * 60)

    from megatron.core.quantization.cakld import CAKLDLoss

    torch.manual_seed(42)

    vocab_size = 500
    hidden_size = 64

    # Simple model
    model = nn.Sequential(
        nn.Embedding(vocab_size, hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, vocab_size),
    ).cuda()

    # CAKLD
    cakld_fn = CAKLDLoss(temperature=2.0, alpha=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # "Teacher" logits (fixed)
    teacher_logits = torch.randn(2, 16, vocab_size, device='cuda')

    losses = []
    for step in range(10):
        optimizer.zero_grad()

        input_ids = torch.randint(0, vocab_size, (2, 16), device='cuda')
        student_logits = model(input_ids)

        # CAKLD loss
        cakld_loss = cakld_fn(student_logits, teacher_logits)

        # LM loss
        lm_loss = F.cross_entropy(student_logits.view(-1, vocab_size), input_ids.view(-1))

        total_loss = lm_loss + 0.1 * cakld_loss

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        if step % 5 == 0:
            print(f"  Step {step}: loss={total_loss.item():.4f}, cakld={cakld_loss.item():.4f}")

    # Verify
    all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
    loss_decreased = losses[-1] < losses[0]

    if all_finite and loss_decreased:
        print(f"\n  [PASS] CAKLD integration: loss decreases, all finite")
        return True
    else:
        print(f"\n  [FAIL] CAKLD integration: issues detected")
        return False


def test_memory_stability():
    """Test 4: Memory stability over extended training."""
    print("\n" + "=" * 60)
    print("Test 4: Memory Stability")
    print("=" * 60)

    from megatron.core.quantization.megatron_qat import QuantConfig, MegatronSeq2BitQuantizer

    torch.manual_seed(42)

    # Model
    vocab_size = 1000
    hidden_size = 128
    num_layers = 4

    model = TinyGPT(vocab_size, hidden_size, 4, num_layers).cuda()

    # Quantizers
    config = QuantConfig(quant_type="seq2bit", group_size=64)
    quantizers = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            q = MegatronSeq2BitQuantizer(module.weight, config, prefix=name)
            q.use_weight_quant = True
            quantizers[name] = q

    # Optimizer
    alpha_params = [q.weight_quantizer.alpha for q in quantizers.values()]
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': alpha_params, 'lr': 1e-3},
    ])

    # Memory tracking
    torch.cuda.empty_cache()
    gc.collect()

    initial_mem = get_memory_mb()
    memories = [initial_mem]

    # Run 50 steps
    for step in range(50):
        optimizer.zero_grad()

        input_ids = torch.randint(0, vocab_size, (4, 32), device='cuda')
        logits = model(input_ids, quantizers)
        loss = F.cross_entropy(logits.view(-1, vocab_size), input_ids.view(-1))

        loss.backward()
        optimizer.step()

        memories.append(get_memory_mb())

    final_mem = memories[-1]
    max_mem = max(memories)
    min_mem = min(memories)

    # Check for memory leak (continuous growth)
    # Allow some growth but not linear increase
    mem_growth = final_mem - initial_mem
    mem_variance = max_mem - min_mem

    print(f"  Initial memory: {initial_mem:.1f}MB")
    print(f"  Final memory: {final_mem:.1f}MB")
    print(f"  Max memory: {max_mem:.1f}MB")
    print(f"  Memory growth: {mem_growth:.1f}MB")
    print(f"  Memory variance: {mem_variance:.1f}MB")

    # Leak detection: if memory grows continuously, it's a leak
    # Calculate slope of memory over time
    x = torch.arange(len(memories), dtype=torch.float32)
    y = torch.tensor(memories, dtype=torch.float32)
    slope = ((x * y).sum() - x.mean() * y.sum()) / ((x ** 2).sum() - len(x) * x.mean() ** 2)

    print(f"  Memory slope: {slope.item():.3f} MB/step")

    # Slope should be small (< 0.5 MB/step) for no leak
    stable = slope.item() < 0.5

    if stable:
        print(f"\n  [PASS] Memory stability: no leak detected")
        return True
    else:
        print(f"\n  [FAIL] Memory stability: potential leak (slope={slope.item():.3f})")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Level 3: Smoke Test - Tiny GPT with QAT")
    print("=" * 70)

    tests = [
        ("tiny_gpt_training", test_tiny_gpt_training),
        ("kd_integration", test_kd_integration),
        ("cakld_integration", test_cakld_integration),
        ("memory_stability", test_memory_stability),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            # Clear memory between tests
            torch.cuda.empty_cache()
            gc.collect()

            results[name] = test_fn()
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Level 3 Smoke Test")
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