#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_symbol(module_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"cmp_dynamic_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


clamp_ste = _load_symbol(ROOT / "EfficientQAT/core/quantizer/ops.py", "clamp_ste")
round_ste = _load_symbol(ROOT / "EfficientQAT/core/quantizer/ops.py", "round_ste")
fake_quant_ste_seq2bit = _load_symbol(
    ROOT / "EfficientQAT/core/quantizer/kernel/fake_quant.py", "fake_quant_ste_seq2bit"
)


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _seq2bit_ref_fake_quant(x: torch.Tensor, alpha: torch.Tensor, group_size: int) -> torch.Tensor:
    ori_shape = x.shape
    xg = x.reshape(-1, group_size)
    s = clamp_ste(alpha.abs(), 1e-6, 1e4).reshape(-1, 1)
    xn = (xg / s).clamp(-1.0, 1.0)
    q = round_ste((xn + 0.75) / 0.5).clamp(0, 3)
    y = (q * 0.5 - 0.75) * s
    return y.reshape(ori_shape)


def summarize_diff(y_ref: torch.Tensor, y_test: torch.Tensor, rel_eps: float):
    ref = y_ref.float().reshape(-1)
    test = y_test.float().reshape(-1)
    finite = torch.isfinite(ref) & torch.isfinite(test)
    n_bad = int((~finite).sum().item())
    if finite.any():
        ref = ref[finite]
        test = test[finite]
    else:
        return {
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "max_rel": float("nan"),
            "mean_rel": float("nan"),
            "l2_rel": float("nan"),
            "n_bad": n_bad,
        }

    diff = test - ref
    abs_err = diff.abs()
    denom = torch.clamp(ref.abs(), min=rel_eps)
    rel_err = abs_err / denom
    l2_ref = torch.norm(ref)
    l2_diff = torch.norm(diff)
    l2_rel = (l2_diff / l2_ref).item() if l2_ref.item() > 0 else float("nan")
    return {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "max_rel": rel_err.max().item(),
        "mean_rel": rel_err.mean().item(),
        "l2_rel": l2_rel,
        "n_bad": n_bad,
    }


def bench_cuda(fn, iters: int, warmup: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_cpu(fn, iters: int, warmup: int):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser(description="Compare seq2bit reference path vs CUDA seq2bit kernel.")
    parser.add_argument("--dtype", default="fp16", choices=DTYPE_MAP.keys())
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mode", default="both", choices=["forward", "fwd_bwd", "both"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rel-eps", type=float, default=1e-6)
    args = parser.parse_args()

    if args.group_size not in (64, 128, 256):
        raise ValueError("group_size must be 64/128/256 for the CUDA kernel.")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU. Kernel benchmark will be skipped.")
        device = "cpu"

    dtype = DTYPE_MAP[args.dtype]
    torch.manual_seed(args.seed)

    in_features = args.group_size * 8
    out_features = max(1, args.num_groups // 8)
    x_base = torch.randn(out_features, in_features, device=device, dtype=dtype).contiguous()
    alpha_base = torch.rand(out_features * (in_features // args.group_size), 1, device=device, dtype=dtype)
    alpha_base = alpha_base.clamp_min(1e-3).contiguous()

    if device == "cuda":
        x_ref = x_base.detach().clone().requires_grad_(True)
        x_ker = x_base.detach().clone().requires_grad_(True)
        a_ref = alpha_base.detach().clone().requires_grad_(True)
        a_ker = alpha_base.detach().clone().requires_grad_(True)
        grad_out = torch.randn_like(x_ref)

        y_ref = _seq2bit_ref_fake_quant(x_ref, a_ref, args.group_size)
        y_ker = fake_quant_ste_seq2bit(x_ker, a_ker.reshape(-1).contiguous(), args.group_size)
        stats = summarize_diff(y_ref, y_ker, args.rel_eps)
        print("[fwd] y")
        print(
            f"  abs_err: max={stats['max_abs']:.6e}, mean={stats['mean_abs']:.6e}; "
            f"rel_err: max={stats['max_rel']:.6e}, mean={stats['mean_rel']:.6e}; "
            f"l2_rel={stats['l2_rel']:.6e}; n_nonfinite={stats['n_bad']}"
        )

        (y_ref * grad_out).sum().backward()
        (y_ker * grad_out).sum().backward()
        g_x = summarize_diff(x_ref.grad, x_ker.grad, args.rel_eps)
        g_a = summarize_diff(a_ref.grad.reshape(-1), a_ker.grad.reshape(-1), args.rel_eps)
        print("[grad] x")
        print(
            f"  abs_err: max={g_x['max_abs']:.6e}, mean={g_x['mean_abs']:.6e}; "
            f"rel_err: max={g_x['max_rel']:.6e}, mean={g_x['mean_rel']:.6e}; "
            f"l2_rel={g_x['l2_rel']:.6e}; n_nonfinite={g_x['n_bad']}"
        )
        print("[grad] alpha")
        print(
            f"  abs_err: max={g_a['max_abs']:.6e}, mean={g_a['mean_abs']:.6e}; "
            f"rel_err: max={g_a['max_rel']:.6e}, mean={g_a['mean_rel']:.6e}; "
            f"l2_rel={g_a['l2_rel']:.6e}; n_nonfinite={g_a['n_bad']}"
        )

    bench = bench_cuda if device == "cuda" else bench_cpu

    def ref_forward():
        return _seq2bit_ref_fake_quant(x_base, alpha_base, args.group_size)

    def ker_forward():
        return fake_quant_ste_seq2bit(x_base, alpha_base.reshape(-1).contiguous(), args.group_size)

    def ref_fwd_bwd():
        x = x_base.detach().clone().requires_grad_(True)
        a = alpha_base.detach().clone().requires_grad_(True)
        y = _seq2bit_ref_fake_quant(x, a, args.group_size)
        y.mean().backward()

    def ker_fwd_bwd():
        x = x_base.detach().clone().requires_grad_(True)
        a = alpha_base.detach().clone().requires_grad_(True)
        y = fake_quant_ste_seq2bit(x, a.reshape(-1).contiguous(), args.group_size)
        y.mean().backward()

    if args.mode in ("forward", "both"):
        if device == "cuda":
            with torch.no_grad():
                t_ref = bench(ref_forward, args.iters, args.warmup)
                t_ker = bench(ker_forward, args.iters, args.warmup)
            print("Speed (forward only, ms/iter)")
            print(f"  seq2bit ref:    {t_ref:.4f}")
            print(f"  seq2bit kernel: {t_ker:.4f}")
            print(f"  speedup:        {t_ref / t_ker:.2f}x")
        else:
            print("Speed (forward only, ms/iter)")
            print(f"  seq2bit ref:    {bench(ref_forward, args.iters, args.warmup):.4f}")
            print("  seq2bit kernel: skipped (CUDA only)")

    if args.mode in ("fwd_bwd", "both"):
        if device == "cuda":
            t_ref = bench(ref_fwd_bwd, args.iters, args.warmup)
            t_ker = bench(ker_fwd_bwd, args.iters, args.warmup)
            print("Speed (forward+backward, ms/iter)")
            print(f"  seq2bit ref:    {t_ref:.4f}")
            print(f"  seq2bit kernel: {t_ker:.4f}")
            print(f"  speedup:        {t_ref / t_ker:.2f}x")
        else:
            print("Speed (forward+backward, ms/iter)")
            print(f"  seq2bit ref:    {bench(ref_fwd_bwd, args.iters, args.warmup):.4f}")
            print("  seq2bit kernel: skipped (CUDA only)")


if __name__ == "__main__":
    main()
