import torch
import torch.nn as nn
import sys
import os
import time

# Ensure the package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from EfficientQAT.core.quantizer.config import QuantConfig
from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear, reinit_quant_params
from EfficientQAT.core.linear.int_quant_linear_infra import IntQuantLinearInfra
from EfficientQAT.core.linear.kernel import int_matmul_backend

def benchmark_models(baseline, optimized, input_tensor, grad_output, steps=100, label=""):
    print(f"\n--- Benchmarking: {label} ---")
    
    # --- Correctness Check ---
    print("Checking correctness...")
    # Forward
    out_base = baseline(input_tensor)
    out_opt = optimized(input_tensor)
    diff = (out_base - out_opt).abs().max().item()
    print(f"Forward Max Diff: {diff:.6e}")
    
    # Backward
    baseline.zero_grad()
    optimized.zero_grad()
    if input_tensor.grad is not None: input_tensor.grad.zero_()
    
    out_base.backward(grad_output, retain_graph=True)
    grad_base_w = baseline.weight.grad.clone() if baseline.weight.grad is not None else None
    
    # Reset input grad for next
    if input_tensor.grad is not None: input_tensor.grad.zero_()
    
    out_opt.backward(grad_output)
    grad_opt_w = optimized.weight.grad.clone() if optimized.weight.grad is not None else None
    
    if grad_base_w is not None and grad_opt_w is not None:
        diff_w = (grad_base_w - grad_opt_w).abs().max().item()
        print(f"Grad Weight Max Diff: {diff_w:.6e}")
    else:
        print("Skipping grad weight check (None)")

    # --- Performance ---
    print(f"Running {steps} steps for timing...")
    
    def measure(model, name):
        # Warmup
        for _ in range(10):
            o = model(input_tensor)
            o.backward(grad_output)
            model.zero_grad()
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(steps):
            o = model(input_tensor)
            o.backward(grad_output)
            model.zero_grad()
        torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / steps * 1000 # ms
        print(f"{name}: {avg_time:.3f} ms / step")
        return avg_time

    t_base = measure(baseline, "Baseline (IntQuantLinear)")
    t_opt = measure(optimized, "Optimized (IntQuantLinearInfra [Triton Kernel])")
    
    print(f"Speedup: {t_base / t_opt:.2f}x")


def run_test():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    n_bits = 4
    group_size = 128
    in_features = 4096
    out_features = 4096 # Square for simplicity
    batch_size = 16
    
    device = torch.device("cuda")
    
    print(f"Config: {in_features}x{out_features}, {n_bits}bit, BS={batch_size}")

    # 1. Setup Baseline
    config = QuantConfig(n_bits=n_bits, group_size=group_size, quant_type="uniform_affine")
    baseline = IntQuantLinear(in_features, out_features, bias=True, config=config).to(device)
    baseline.use_weight_quant = True
    
    # Init Baseline Params
    with torch.no_grad():
        reinit_quant_params(nn.Sequential(baseline))
    
    # 2. Setup Optimized (From Baseline)
    optimized = IntQuantLinearInfra.from_qat(baseline).to(device)
    # Use Triton Backend for testing
    optimized.kernel_backend = int_matmul_backend 
    
    # --- Check Params Equality ---
    s_base, z_base = baseline.weight_quantizer.cal_qparams(
        baseline.weight_quantizer.scale, baseline.weight_quantizer.zero_point
    )
    s_opt = optimized.scales
    z_opt = optimized.qzeros

    # Base quantizer returns [num_groups, 1]; reshape to [out_features, groups_per_row]
    s_base = s_base.view(out_features, -1)
    z_base = z_base.view(out_features, -1)
    
    diff_s = (s_base - s_opt).abs().max().item()
    diff_z = (z_base - z_opt).abs().max().item()
    print(f"Scales Max Diff: {diff_s:.6e}")
    print(f"Zeros Max Diff: {diff_z:.6e}")
    
    # 3. Inputs
    # Use float32 to match default JIT logic if needed, but layers are float32 by default
    input_tensor = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    grad_output = torch.randn(batch_size, out_features, device=device)
    
    benchmark_models(baseline, optimized, input_tensor, grad_output, label="Comparison")

if __name__ == "__main__":
    run_test()
