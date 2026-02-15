import contextlib
import os
import torch
from torch.utils.cpp_extension import load

try:
    from torch.cuda import nvtx as _nvtx
except Exception:
    _nvtx = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_ext = load(
    name="fake_quant_ext",
    sources=[os.path.join(_THIS_DIR, "fake_quant_ext.cu")],
    extra_cuda_cflags=["-O3",
                    "--use_fast_math",
                    "-lineinfo", # for better debugging
                    "-gencode arch=compute_89,code=sm_89", # adjust according to your GPU
                    "-std=c++17"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)

@contextlib.contextmanager
def _nvtx_range(name: str):
    if _nvtx is None:
        yield
        return
    if hasattr(_nvtx, "range_start") and hasattr(_nvtx, "range_end"):
        handle = _nvtx.range_start(name)
        try:
            yield
        finally:
            _nvtx.range_end(handle)
    else:
        _nvtx.range_push(name)
        try:
            yield
        finally:
            _nvtx.range_pop()

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zp, qmin: int, qmax: int, group_size: int):
        """
        x: contiguous CUDA tensor, dtype fp16/bf16/fp32
        scale: raw scale (pre-softplus), [N_groups] (any shape) contiguous CUDA tensor, dtype fp16/bf16/fp32
        zp: raw zero_point, None or [N_groups] contiguous CUDA tensor, dtype fp16/bf16/fp32
        """
        assert x.is_cuda and x.is_contiguous()
        assert scale.is_cuda and scale.is_contiguous()
        assert group_size in (64, 128, 256)

        assert x.numel() % group_size == 0
        N_groups = x.numel() // group_size
        assert scale.numel() == N_groups

        if zp is not None:
            assert zp.is_cuda and zp.is_contiguous()
            assert zp.dtype in (torch.float16, torch.bfloat16, torch.float32)
            assert zp.numel() == N_groups

        with _nvtx_range("fake_quant_fwd"):
            y = _ext.fwd(x, scale, zp, int(qmin), int(qmax), int(group_size))

        # save for backward
        ctx.qmin = int(qmin)
        ctx.qmax = int(qmax)
        ctx.group_size = int(group_size)
        ctx.save_for_backward(x, scale, zp if zp is not None else torch.empty(0, device=x.device, dtype=torch.float32))

        return y

    @staticmethod
    def backward(ctx, dy):
        x, scale, zp_buf = ctx.saved_tensors
        qmin, qmax, G = ctx.qmin, ctx.qmax, ctx.group_size

        dy2 = dy.contiguous()

        zp = None
        if zp_buf.numel() != 0:
            zp = zp_buf

        with _nvtx_range("fake_quant_bwd"):
            dx, dscale, dzp = _ext.bwd(x, dy2, scale, zp, int(qmin), int(qmax), int(G))

        # dzp: empty tensor if zp is None
        dzp_out = None if zp is None else dzp

        return dx, dscale, dzp_out, None, None, None

def fake_quant_ste(x: torch.Tensor,
                   scale: torch.Tensor,
                   zp: torch.Tensor | None,
                   qmin: int,
                   qmax: int,
                   group_size: int) -> torch.Tensor:
    return FakeQuantSTE.apply(x, scale, zp, int(qmin), int(qmax), int(group_size))


class FakeQuantSTESeq2Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, group_size: int):
        assert x.is_cuda and x.is_contiguous()
        assert alpha.is_cuda and alpha.is_contiguous()
        assert alpha.dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert group_size in (64, 128, 256)
        assert x.numel() % group_size == 0
        assert alpha.numel() == x.numel() // group_size

        with _nvtx_range("fake_quant_seq2bit_fwd"):
            y = _ext.fake_quant_ste_seq2bit_fwd_cuda(x, alpha, int(group_size))

        ctx.group_size = int(group_size)
        ctx.save_for_backward(x, alpha)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, alpha = ctx.saved_tensors
        G = ctx.group_size
        dy2 = dy.contiguous()

        with _nvtx_range("fake_quant_seq2bit_bwd"):
            dx, dalpha = _ext.fake_quant_ste_seq2bit_bwd_cuda(x, dy2, alpha, int(G))

        return dx, dalpha, None


def fake_quant_ste_seq2bit(
    x: torch.Tensor,
    alpha: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return FakeQuantSTESeq2Bit.apply(x, alpha, int(group_size))


def _make_qparams(x: torch.Tensor, qmin: int, qmax: int):
    x2 = x.detach().reshape(x.shape[0], -1)
    scale = x2.abs().amax(dim=1) / max(int(qmax), 1)
    scale = scale.clamp(min=1e-6).contiguous()
    zp = torch.zeros_like(scale, dtype=torch.float32)
    return scale, zp


if __name__ == "__main__":
    """
    Example NCU commands:
    ncu --set full --target-processes all --nvtx --nvtx-include "fake_quant_fwd" \
        -o tmp/ncu_fake_quant_fwd \
        python EfficientQAT/core/quantizer/kernel/fake_quant.py
    
    ncu --set full --target-processes all --nvtx --nvtx-include "fake_quant_bwd" \
        -o tmp/ncu_fake_quant_bwd \
        python EfficientQAT/core/quantizer/kernel/fake_quant.py

    """
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Quick fake_quant kernel test with NVTX ranges.")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=4096)
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the fake_quant kernel test.")
    if args.group_size not in (64, 128, 256):
        raise SystemExit("group_size must be 64/128/256 for the CUDA kernel.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = "cuda"

    qmin = 0
    qmax = (1 << int(args.n_bits)) - 1

    x = torch.randn(args.num_groups, args.group_size, device=device, dtype=dtype, requires_grad=True).contiguous()
    scale, zp = _make_qparams(x, qmin, qmax)

    def run_fwd():
        return fake_quant_ste(x, scale, zp, qmin, qmax, args.group_size)

    def run_fwd_bwd():
        if x.grad is not None:
            x.grad.zero_()
        y = run_fwd()
        y.mean().backward()

    for _ in range(args.warmup):
        run_fwd_bwd()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        run_fwd_bwd()
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"avg fwd+bwd time: {(end - start) * 1000.0 / args.iters:.3f} ms")
