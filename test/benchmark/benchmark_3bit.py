#!/usr/bin/env python3
"""
Benchmark script for 3-bit dequantization kernel.
"""

import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear
from EfficientQAT.core.linear.q_linear_triton_kernels import dequant


def benchmark_dequant(bits, in_features, out_features, group_size, num_warmup=10, num_iters=100, use_v2=True):
    """Benchmark dequantization kernel."""
    
    # Create quantized linear
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    torch.manual_seed(42)
    linear.weight.data = torch.randn(out_features, in_features) * 0.1
    
    qlinear = TritonV2QuantLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
    )
    
    num_groups = in_features // group_size
    scales = torch.ones(out_features, num_groups) * 0.1
    zeros = torch.ones(out_features, num_groups) * (2 ** (bits - 1))
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    
    qlinear.pack(linear, scales, zeros, g_idx)
    qlinear.post_init()
    
    # Move to CUDA
    qlinear.qweight = qlinear.qweight.cuda()
    qlinear.qzeros = qlinear.qzeros.cuda()
    qlinear.scales = qlinear.scales.cuda()
    qlinear.g_idx = qlinear.g_idx.cuda()
    
    # Warmup
    for _ in range(num_warmup):
        _ = dequant(
            qlinear.qweight,
            qlinear.scales,
            qlinear.qzeros,
            qlinear.g_idx,
            bits=bits,
            pack_bits=32,
            maxq=2**bits - 1,
            sym=False,
            use_v2=use_v2,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = dequant(
            qlinear.qweight,
            qlinear.scales,
            qlinear.qzeros,
            qlinear.g_idx,
            bits=bits,
            pack_bits=32,
            maxq=2**bits - 1,
            sym=False,
            use_v2=use_v2,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_iters * 1000
    
    # Calculate throughput
    num_elements = in_features * out_features
    throughput_gops = num_elements / avg_time_ms / 1e6  # G elements/s
    
    return avg_time_ms, throughput_gops


def benchmark_forward(bits, in_features, out_features, group_size, batch_size, seq_len, num_warmup=10, num_iters=100):
    """Benchmark full forward pass."""
    
    linear = torch.nn.Linear(in_features, out_features, bias=True)
    torch.manual_seed(42)
    linear.weight.data = torch.randn(out_features, in_features) * 0.1
    linear.bias.data = torch.randn(out_features) * 0.01
    
    qlinear = TritonV2QuantLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )
    
    num_groups = in_features // group_size
    scales = torch.ones(out_features, num_groups) * 0.1
    zeros = torch.ones(out_features, num_groups) * (2 ** (bits - 1))
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    
    qlinear.pack(linear, scales, zeros, g_idx)
    qlinear.post_init()
    
    # Move to CUDA
    qlinear.qweight = qlinear.qweight.cuda()
    qlinear.qzeros = qlinear.qzeros.cuda()
    qlinear.scales = qlinear.scales.cuda()
    qlinear.g_idx = qlinear.g_idx.cuda()
    qlinear.bias = qlinear.bias.cuda()
    qlinear.wf_unsqueeze_zero = qlinear.wf_unsqueeze_zero.cuda()
    qlinear.wf_unsqueeze_neg_one = qlinear.wf_unsqueeze_neg_one.cuda()
    
    x = torch.randn(batch_size, seq_len, in_features, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = qlinear(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = qlinear(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_iters * 1000
    
    # Calculate TFLOPS (2 * M * N * K for matmul)
    flops = 2 * batch_size * seq_len * in_features * out_features
    tflops = flops / avg_time_ms / 1e9
    
    return avg_time_ms, tflops


def main():
    print("=" * 100)
    print("3-bit Quantization Kernel Benchmark (v1 vs v2)")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Test configurations (typical LLM layer sizes)
    configs = [
        # (in_features, out_features, group_size)
        (4096, 4096, 128),   # Llama-7B attention
        (4096, 11008, 128),  # Llama-7B MLP up/gate
        (11008, 4096, 128),  # Llama-7B MLP down
        (896, 896, 64),      # Qwen2.5-0.5B attention
        (896, 4864, 64),     # Qwen2.5-0.5B MLP
    ]
    
    print("Dequantization Kernel Benchmark:")
    print("-" * 100)
    print(f"{'Config':<25} {'4-bit (ms)':<12} {'3-bit v1':<12} {'3-bit v2':<12} {'v1/4bit':<10} {'v2/4bit':<10} {'v2/v1':<10}")
    print("-" * 100)
    
    for in_f, out_f, gs in configs:
        results = {}
        # 4-bit baseline
        try:
            time_ms, _ = benchmark_dequant(4, in_f, out_f, gs)
            results['4bit'] = time_ms
        except Exception as e:
            results['4bit'] = float('nan')
            print(f"Error with 4-bit: {e}")
        
        # 3-bit v1
        try:
            time_ms, _ = benchmark_dequant(3, in_f, out_f, gs, use_v2=False)
            results['3bit_v1'] = time_ms
        except Exception as e:
            results['3bit_v1'] = float('nan')
            print(f"Error with 3-bit v1: {e}")
        
        # 3-bit v2
        try:
            time_ms, _ = benchmark_dequant(3, in_f, out_f, gs, use_v2=True)
            results['3bit_v2'] = time_ms
        except Exception as e:
            results['3bit_v2'] = float('nan')
            print(f"Error with 3-bit v2: {e}")
        
        ratio_v1 = results['3bit_v1'] / results['4bit'] if results['4bit'] > 0 else float('nan')
        ratio_v2 = results['3bit_v2'] / results['4bit'] if results['4bit'] > 0 else float('nan')
        ratio_v2_v1 = results['3bit_v2'] / results['3bit_v1'] if results['3bit_v1'] > 0 else float('nan')
        
        print(f"{in_f}x{out_f} g{gs:<6} {results['4bit']:<12.4f} {results['3bit_v1']:<12.4f} {results['3bit_v2']:<12.4f} {ratio_v1:<10.2f}x {ratio_v2:<10.2f}x {ratio_v2_v1:<10.2f}x")
    
    print()
    print("Forward Pass Benchmark (batch=1, seq=2048):")
    print("-" * 100)
    print(f"{'Config':<25} {'4-bit (ms)':<12} {'3-bit (ms)':<12} {'Ratio 3/4':<10}")
    print("-" * 100)
    
    batch_size = 1
    seq_len = 2048
    
    for in_f, out_f, gs in configs:
        results = {}
        for bits in [3, 4]:
            try:
                time_ms, _ = benchmark_forward(bits, in_f, out_f, gs, batch_size, seq_len)
                results[bits] = time_ms
            except Exception as e:
                results[bits] = float('nan')
        
        ratio = results[3] / results[4] if results[4] > 0 else float('nan')
        print(f"{in_f}x{out_f} g{gs:<6} {results[4]:<12.4f} {results[3]:<12.4f} {ratio:<10.2f}x")


if __name__ == "__main__":
    main()
