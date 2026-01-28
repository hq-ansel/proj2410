#!/usr/bin/env python3
import argparse
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from EfficientQAT.core.quantizer.uniform_affine import UniformAffineQuantizer
from EfficientQAT.core.quantizer.config import QuantConfig
from EfficientQAT.core.quantizer.kernel.fake_quant import fake_quant_ste


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def make_quantizer(weight: torch.Tensor, n_bits: int, group_size: int, clamp_method: str) -> UniformAffineQuantizer:
    config = QuantConfig(
        n_bits=n_bits,
        group_size=group_size,
        clamp_method=clamp_method,
        is_tracking=False,
        stat_quant=False,
    )
    return UniformAffineQuantizer(prefix="cmp", weight=weight, config=config).to(weight.device)


def get_kernel_qparams(quantizer: UniformAffineQuantizer, *, zp_fp32: bool):
    scale = quantizer.scale.contiguous()
    zp = None
    if quantizer.zero_point is not None:
        zp = quantizer.zero_point.contiguous()
        if zp_fp32:
            zp = zp.float()
    return scale, zp


def _format_quantiles(vals, qs, max_val):
    parts = [f"p{int(q * 100)}={v:.6e}" if q < 0.999 else f"p99.9={v:.6e}" for q, v in zip(qs, vals)]
    parts.append(f"max={max_val:.6e}")
    return ", ".join(parts)


def _tensor_quantiles(x: torch.Tensor, qs):
    flat = x.reshape(-1)
    if flat.numel() == 0:
        return None
    q = torch.tensor(qs, device=flat.device, dtype=torch.float32)
    vals = torch.quantile(flat.float(), q)
    return vals.cpu().tolist()


def _ulp_diff_quantiles(y_ref: torch.Tensor, y_test: torch.Tensor, qs):
    try:
        import numpy as np
    except Exception:
        return None, "numpy-not-available"

    ref_cpu = y_ref.detach().cpu().contiguous()
    test_cpu = y_test.detach().cpu().contiguous()
    if test_cpu.dtype != ref_cpu.dtype:
        test_cpu = test_cpu.to(ref_cpu.dtype)
    dtype = ref_cpu.dtype

    ref_fp32 = ref_cpu.float().numpy()
    test_fp32 = test_cpu.float().numpy()
    valid = np.isfinite(ref_fp32) & np.isfinite(test_fp32)
    if not valid.any():
        return None, "no-finite-values"

    if dtype == torch.float32:
        ref_bits = ref_cpu.numpy().view(np.int32)
        test_bits = test_cpu.numpy().view(np.int32)
        ref_bits = ref_bits[valid]
        test_bits = test_bits[valid]
        ordered_ref = ref_bits.astype(np.int64)
        ordered_test = test_bits.astype(np.int64)
        mask = ordered_ref < 0
        ordered_ref[mask] = 0x80000000 - ordered_ref[mask]
        mask = ordered_test < 0
        ordered_test[mask] = 0x80000000 - ordered_test[mask]
    elif dtype == torch.float16:
        ref_bits = ref_cpu.numpy().view(np.int16)
        test_bits = test_cpu.numpy().view(np.int16)
        ref_bits = ref_bits[valid]
        test_bits = test_bits[valid]
        ordered_ref = ref_bits.astype(np.int32)
        ordered_test = test_bits.astype(np.int32)
        mask = ordered_ref < 0
        ordered_ref[mask] = 0x8000 - ordered_ref[mask]
        mask = ordered_test < 0
        ordered_test[mask] = 0x8000 - ordered_test[mask]
    elif dtype == torch.bfloat16:
        ref_bits32 = ref_cpu.float().numpy().view(np.int32)
        test_bits32 = test_cpu.float().numpy().view(np.int32)
        ref_bits = (ref_bits32 >> 16).astype(np.int16)
        test_bits = (test_bits32 >> 16).astype(np.int16)
        ref_bits = ref_bits[valid]
        test_bits = test_bits[valid]
        ordered_ref = ref_bits.astype(np.int32)
        ordered_test = test_bits.astype(np.int32)
        mask = ordered_ref < 0
        ordered_ref[mask] = 0x8000 - ordered_ref[mask]
        mask = ordered_test < 0
        ordered_test[mask] = 0x8000 - ordered_test[mask]
    else:
        return None, f"unsupported-dtype-{dtype}"

    ulp = np.abs(ordered_ref.astype(np.int64) - ordered_test.astype(np.int64))
    max_val = float(ulp.max()) if ulp.size else float("nan")
    q_vals = np.quantile(ulp, qs).tolist() if ulp.size else None
    return (q_vals, max_val), None


def summarize_diff(y_ref: torch.Tensor, y_test: torch.Tensor, rel_eps: float):
    diff = (y_test - y_ref).float()
    abs_err = diff.abs()
    denom = torch.clamp(y_ref.abs().float(), min=rel_eps)
    rel_err = abs_err / denom

    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rms_abs = torch.sqrt((diff * diff).mean()).item()

    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()
    rms_rel = torch.sqrt((rel_err * rel_err).mean()).item()

    qs = [0.5, 0.9, 0.99, 0.999]
    abs_q = _tensor_quantiles(abs_err, qs)
    rel_q = _tensor_quantiles(rel_err, qs)

    ref_flat = y_ref.float().reshape(-1)
    test_flat = y_test.float().reshape(-1)
    diff_flat = diff.reshape(-1)
    ref_norm = torch.norm(ref_flat)
    diff_norm = torch.norm(diff_flat)
    l2_rel = (diff_norm / ref_norm).item() if ref_norm.item() > 0 else float("nan")
    test_norm = torch.norm(test_flat)
    denom = (ref_norm * test_norm).item()
    cos_sim = (torch.dot(ref_flat, test_flat).item() / denom) if denom > 0 else float("nan")

    ulp_stats, ulp_note = _ulp_diff_quantiles(y_ref, y_test, qs)

    return {
        "abs": (max_abs, mean_abs, rms_abs, abs_q),
        "rel": (max_rel, mean_rel, rms_rel, rel_q),
        "tensor": (l2_rel, cos_sim),
        "ulp": (ulp_stats, ulp_note),
        "qs": qs,
    }


def _manual_forward(x: torch.Tensor,
                    scale_raw: torch.Tensor,
                    zp_raw: torch.Tensor | None,
                    qmin: int,
                    qmax: int,
                    group_size: int,
                    *,
                    fp32: bool):
    min_scale = 1e-5
    max_scale = 1e4
    if fp32:
        x = x.float()
        scale_raw = scale_raw.float()
        if zp_raw is not None:
            zp_raw = zp_raw.float()

    xg = x.reshape(-1, group_size)
    scale_raw = scale_raw.reshape(-1, 1)
    zpf = None
    if zp_raw is not None:
        zpf = torch.round(zp_raw.reshape(-1, 1))
        zpf = torch.clamp(zpf, qmin, qmax)

    sign = torch.where(scale_raw >= 0, torch.ones_like(scale_raw), -torch.ones_like(scale_raw))
    s = torch.clamp(scale_raw.abs(), min_scale, max_scale) * sign
    inv_s = 1.0 / s
    if zpf is None:
        zpf = torch.zeros_like(s)

    u = xg * inv_s + zpf
    q = torch.round(u)
    q = torch.clamp(q, qmin, qmax)
    y = (q - zpf) * s
    return y.reshape_as(x)


def _manual_dscale(xg: torch.Tensor,
                   grad_outg: torch.Tensor,
                   scale_raw: torch.Tensor,
                   zp_raw: torch.Tensor | None,
                   qmin: int,
                   qmax: int,
                   *,
                   fp32: bool):
    min_scale = 1e-5
    max_scale = 1e4
    if fp32:
        xg = xg.float()
        grad_outg = grad_outg.float()
        scale_raw = scale_raw.float()
        if zp_raw is not None:
            zp_raw = zp_raw.float()

    scale_raw = scale_raw.reshape(-1, 1)
    zpf = None
    if zp_raw is not None:
        zpf = torch.round(zp_raw.reshape(-1, 1))
        zpf = torch.clamp(zpf, qmin, qmax)

    sign = torch.where(scale_raw >= 0, torch.ones_like(scale_raw), -torch.ones_like(scale_raw))
    s = torch.clamp(scale_raw.abs(), min_scale, max_scale) * sign
    inv_s = 1.0 / s
    if zpf is None:
        zpf = torch.zeros_like(s)
    u = xg * inv_s + zpf
    q_unclamped = torch.round(u)
    mask = (q_unclamped >= qmin) & (q_unclamped <= qmax)
    q = torch.clamp(q_unclamped, qmin, qmax)
    term = (q - zpf) - mask.float() * (xg * inv_s)
    dscale = (grad_outg * term).sum(dim=1, keepdim=True)
    return dscale, mask, s


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
    parser = argparse.ArgumentParser(description="Compare UniformAffineQuantizer vs CUDA fake_quant kernel.")
    parser.add_argument("--dtype", default="fp16", choices=DTYPE_MAP.keys())
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=4096)
    parser.add_argument("--clamp-method", default="STE", choices=["STE", "MAD"])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mode", default="both", choices=["forward", "fwd_bwd", "both"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rel-eps", type=float, default=1e-6)
    parser.add_argument("--zp-fp32", action="store_true",
                        help="Force zero_point to fp32 for both ref/kernel paths (debug).")
    args = parser.parse_args()

    if args.group_size not in (64, 128, 256):
        raise ValueError("group_size must be 64/128/256 for the CUDA kernel.")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU. Kernel test will be skipped.")
        device = "cpu"

    torch.manual_seed(args.seed)
    dtype = DTYPE_MAP[args.dtype]

    shape = (128, 1024)

    weight = torch.randn(shape[0], shape[1], device=device, dtype=dtype)
    quantizer = make_quantizer(weight, args.n_bits, args.group_size, args.clamp_method)

    def clone_leaf(t: torch.Tensor, *, requires_grad: bool):
        # 让它成为 leaf，避免 share graph
        out = t.detach().clone()
        out.requires_grad_(requires_grad)
        return out

    # 用 weight 作为输入，避免和量化参数来源不一致
    x = clone_leaf(weight, requires_grad=True).contiguous()
    # 统计x的p50 p90 p99 p99.9
    print(f"x statistics: p50={torch.quantile(x.float(), 0.5):.6e}, p90={torch.quantile(x.float(), 0.9):.6e}, p99={torch.quantile(x.float(), 0.99):.6e}, p99.9={torch.quantile(x.float(), 0.999):.6e}")
    print(f"x min abs = {torch.min(x.abs()):.6e}")
    # Accuracy compare (forward only)
    if device == "cuda":
        torch.cuda.synchronize()

        # -------------------------
        # Prepare "same" initial values
        # -------------------------
        # x: always compare
        x_ref = clone_leaf(weight, requires_grad=True)
        x_ker = clone_leaf(weight, requires_grad=True)

        # get qparams from quantizer
        # 你现在的 get_kernel_qparams(quantizer) 可能返回已经 round/clip 过的 scale/zp
        # 如果要比较 scale/zp 的梯度，请保证这里拿到的是 float 且可导的版本
        scale0, zp0 = get_kernel_qparams(quantizer, zp_fp32=args.zp_fp32)

        # scale grad compare (float, leaf)
        scale_ref = clone_leaf(scale0, requires_grad=True)
        scale_ker = clone_leaf(scale0, requires_grad=True)

        # zp grad compare:
        # - 如果 zp0 是 int / long / 已 round：zp_ref.grad 会是 None（不可导）
        # - 如果你希望比较 zp.grad：需要让 zp0 是 float（未round），并在 ref/kernel 两侧都用 STE 方式处理 round
        zp_requires_grad = (zp0.is_floating_point())
        zp_ref = clone_leaf(zp0, requires_grad=zp_requires_grad)
        zp_ker = clone_leaf(zp0, requires_grad=zp_requires_grad)

        # 固定上游梯度（避免不同 loss 形态导致差异）
        grad_out = torch.randn_like(x_ref)

        # -------------------------
        # REF path (quantizer.fake_quant)
        # -------------------------
        # 如果 quantizer.fake_quant 内部自己算 qparams，你就没法把 scale_ref/zp_ref 注入进去，
        # 那 scale/zp 的梯度比较在逻辑上是不成立的（因为 ref 路径的 scale/zp 不是同一个变量）。
        #
        # 建议你在 quantizer 里加一个可选接口：fake_quant_with_qparams(x, scale, zp)
        # 下面先按"你已经能注入 scale/zp"来写：
        # scale_ref=scale_ref.unsqueeze(-1)
        # zp_ref = zp_ref.unsqueeze(-1)
        y_ref = quantizer.fake_quant(x_ref, scale_ref, zp_ref)
        loss_ref = (y_ref * grad_out).sum()
        loss_ref.backward()

        # -------------------------
        # KERNEL path
        # -------------------------
        y_ker = fake_quant_ste(x_ker, scale_ker, zp_ker,
                            quantizer.qmin, quantizer.qmax, args.group_size)
        fwd_stats = summarize_diff(y_ref, y_ker, args.rel_eps)
        max_abs, mean_abs, rms_abs, abs_q = fwd_stats["abs"]
        max_rel, mean_rel, rms_rel, rel_q = fwd_stats["rel"]
        l2_rel, cos_sim = fwd_stats["tensor"]
        print("[fwd] y")
        print(f"  abs_err: max={max_abs:.6e}, mean={mean_abs:.6e}, rms={rms_abs:.6e}")
        print(f"  rel_err: max={max_rel:.6e}, mean={mean_rel:.6e}, rms={rms_rel:.6e}")
        print(f"  tensor: l2_rel={l2_rel:.6e}, cosine={cos_sim:.6e}")
        y_manual_fp16 = _manual_forward(
            x_ref, scale_ref, zp_ref if zp0 is not None else None,
            quantizer.qmin, quantizer.qmax, args.group_size, fp32=False
        )
        y_manual_fp32 = _manual_forward(
            x_ref, scale_ref, zp_ref if zp0 is not None else None,
            quantizer.qmin, quantizer.qmax, args.group_size, fp32=True
        )
        print("[fwd-debug]")
        def _brief(name, a, b):
            stats = summarize_diff(a, b, args.rel_eps)
            l2_rel, cos_sim = stats["tensor"]
            print(f"  {name}: l2_rel={l2_rel:.6e}, cosine={cos_sim:.6e}")
        _brief("manual(fp16) vs ref", y_manual_fp16, y_ref)
        _brief("manual(fp32) vs ref", y_manual_fp32, y_ref)
        _brief("manual(fp32) vs kernel", y_manual_fp32, y_ker)
        loss_ker = (y_ker * grad_out).sum()
        loss_ker.backward()

        # -------------------------
        # Compare grads
        # -------------------------
        def compare(name, a, b):
            if a is None or b is None:
                print(f"{name}: grad compare skipped (a is {a is None}, b is {b is None})")
                return
            stats = summarize_diff(a, b, args.rel_eps)
            max_abs, mean_abs, rms_abs, abs_q = stats["abs"]
            max_rel, mean_rel, rms_rel, rel_q = stats["rel"]
            l2_rel, cos_sim = stats["tensor"]
            print(f"[grad] {name}")
            print(f"  abs_err: max={max_abs:.6e}, mean={mean_abs:.6e}, rms={rms_abs:.6e}")
            print(f"  rel_err: max={max_rel:.6e}, mean={mean_rel:.6e}, rms={rms_rel:.6e}")
            print(f"  tensor: l2_rel={l2_rel:.6e}, cosine={cos_sim:.6e}")

        compare("x.grad", x_ref.grad, x_ker.grad)
        compare("scale.grad", scale_ref.grad, scale_ker.grad)
        compare("zp.grad", zp_ref.grad, zp_ker.grad)

        # -------------------------
        # Analyze scale.grad (manual formula vs autograd)
        # -------------------------
        xg = x_ref.reshape(-1, args.group_size)
        grad_outg = grad_out.reshape(-1, args.group_size)
        dscale_fp16, mask_fp16, s_fp16 = _manual_dscale(
            xg, grad_outg, scale_ref, zp_ref if zp0 is not None else None,
            quantizer.qmin, quantizer.qmax, fp32=False
        )
        dscale_fp32, mask_fp32, s_fp32 = _manual_dscale(
            xg, grad_outg, scale_ref, zp_ref if zp0 is not None else None,
            quantizer.qmin, quantizer.qmax, fp32=True
        )

        def _stats(t: torch.Tensor):
            flat = t.reshape(-1).float()
            qs = torch.tensor([0.0, 0.5, 0.99, 1.0], device=flat.device)
            vals = torch.quantile(flat, qs).cpu().tolist()
            return vals

        mask_mean = mask_fp32.float().mean().item()
        per_group_in = mask_fp32.float().mean(dim=1)
        per_q = torch.tensor([0.5, 0.9, 0.99], device=per_group_in.device)
        per_vals = torch.quantile(per_group_in, per_q).cpu().tolist()

        s_min, s_med, s_p99, s_max = _stats(s_fp32)
        raw_min, raw_med, raw_p99, raw_max = _stats(scale_ref)

        print("[scale-debug]")
        print(f"  raw_scale: min={raw_min:.6e}, p50={raw_med:.6e}, p99={raw_p99:.6e}, max={raw_max:.6e}")
        print(f"  s=clamp(abs(scale),1e-5,1e4)*sign: min={s_min:.6e}, p50={s_med:.6e}, p99={s_p99:.6e}, max={s_max:.6e}")
        print(f"  mask(in-range): mean={mask_mean:.6e}, p50={per_vals[0]:.6e}, p90={per_vals[1]:.6e}, p99={per_vals[2]:.6e}")

        def _brief_diff(name: str, a: torch.Tensor, b: torch.Tensor):
            stats = summarize_diff(a, b, args.rel_eps)
            l2_rel, cos_sim = stats["tensor"]
            print(f"  {name}: l2_rel={l2_rel:.6e}, cosine={cos_sim:.6e}")

        _brief_diff("manual(fp32) vs ref.grad", dscale_fp32, scale_ref.grad)
        _brief_diff("manual(fp32) vs kernel.grad", dscale_fp32, scale_ker.grad)
        _brief_diff("manual(fp16) vs manual(fp32)", dscale_fp16, dscale_fp32)

    else:
        print("Grad compare skipped (kernel requires CUDA).")

    # Speed compare
    if device == "cuda":
        bench = bench_cuda
    else:
        bench = bench_cpu

    def py_forward():
        return quantizer.fake_quant(x)

    def kernel_forward():
        scale, zp = get_kernel_qparams(quantizer, zp_fp32=args.zp_fp32)
        return fake_quant_ste(x, scale, zp, quantizer.qmin, quantizer.qmax, args.group_size)

    def py_fwd_bwd():
        if x.grad is not None:
            x.grad.zero_()
        quantizer.zero_grad(set_to_none=True)
        y = quantizer.fake_quant(x)
        y.mean().backward()

    def kernel_fwd_bwd():
        if x.grad is not None:
            x.grad.zero_()
        quantizer.zero_grad(set_to_none=True)
        scale, zp = get_kernel_qparams(quantizer, zp_fp32=args.zp_fp32)
        y = fake_quant_ste(x, scale, zp, quantizer.qmin, quantizer.qmax, args.group_size)
        y.mean().backward()

    if args.mode in ("forward", "both"):
        with torch.no_grad():
            t_py = bench(py_forward, args.iters, args.warmup)
            if device == "cuda":
                t_kernel = bench(kernel_forward, args.iters, args.warmup)
            else:
                t_kernel = float("nan")
        print("Speed (forward only, ms/iter)")
        print(f"  UniformAffineQuantizer: {t_py:.4f}")
        if device == "cuda":
            print(f"  fake_quant kernel:      {t_kernel:.4f}")
            print(f"  speedup:                {t_py / t_kernel:.2f}x")
        else:
            print("  fake_quant kernel:      skipped (CUDA only)")

    if args.mode in ("fwd_bwd", "both"):
        t_py = bench(py_fwd_bwd, args.iters, args.warmup)
        if device == "cuda":
            t_kernel = bench(kernel_fwd_bwd, args.iters, args.warmup)
        else:
            t_kernel = float("nan")
        print("Speed (forward+backward, ms/iter)")
        print(f"  UniformAffineQuantizer: {t_py:.4f}")
        if device == "cuda":
            print(f"  fake_quant kernel:      {t_kernel:.4f}")
            print(f"  speedup:                {t_py / t_kernel:.2f}x")
        else:
            print("  fake_quant kernel:      skipped (CUDA only)")

# python EfficientQAT/core/quantizer/test/compare_uniform_affine_vs_kernel.py --dtype fp16 --group-size 128 --num-groups 4096 --iters 200 --warmup 20
if __name__ == "__main__":
    main()
