import torch
import torch.nn as nn
import argparse
import sys
import os

# Ensure the project root is in PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EfficientQAT.core.linear.int_quant_linear_infra import IntQuantLinearInfra
from EfficientQAT.core.quantizer.config import QuantConfig

def profile_layer(batch_size, in_features, out_features, n_bits, group_size, steps=10, mode="forward"):
    if not torch.cuda.is_available():
        print("CUDA not available, cannot profile.")
        return

    device = torch.device("cuda")
    print(f"Profiling {mode} on {device}")
    
    # Create Configuration
    config = QuantConfig(n_bits=n_bits, group_size=group_size)
    
    # Initialize Layer
    layer = IntQuantLinearInfra(
        in_features=in_features, 
        out_features=out_features, 
        bias=True, 
        config=config
    ).to(device)
    
    # Initialize Inputs
    dtype = torch.float32 
    input_tensor = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(batch_size, out_features, device=device, dtype=dtype)

    # Warmup
    print("Warmup...")
    for _ in range(5):
        output = layer(input_tensor)
        if output.requires_grad:
            output.backward(grad_output)
        layer.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
    
    torch.cuda.synchronize()
        
    print(f"Start Profiling ({steps} steps)...")
    
    if mode == "forward":
        torch.cuda.profiler.start()
        for _ in range(steps):
            output = layer(input_tensor)
        torch.cuda.profiler.stop()
        
    elif mode == "backward":
        # We need to run forward to build graph, but we don't want to profile it if possible.
        # But profiler.start() captures everything.
        # So we run forward OUTSIDE the profiler region for each step?
        # No, backward requires the graph from the specific forward run usually.
        # But we can do:
        # Loop:
        #   Forward (no profile)
        #   Profiler Start
        #   Backward
        #   Profiler Stop
        # This might introduce overhead of starting/stopping profiler many times.
        # Alternatively, we profile both and ignore forward kernels in post-processing?
        # Or just profile the whole loop and accept forward is included?
        # A better way for backward:
        # 1. Forward
        # 2. Profiler Start
        # 3. Backward
        # 4. Profiler Stop
        # Repeat? 
        # ncu handles multiple start/stops fine.
        
        for _ in range(steps):
            output = layer(input_tensor)
            
            torch.cuda.profiler.start()
            output.backward(grad_output)
            torch.cuda.profiler.stop()
            
            layer.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
    
    torch.cuda.synchronize()
    print("Profiling Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=11008)
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "backward"])
    args = parser.parse_args()
    
    profile_layer(args.batch_size, args.in_features, args.out_features, args.n_bits, args.group_size, args.steps, args.mode)
