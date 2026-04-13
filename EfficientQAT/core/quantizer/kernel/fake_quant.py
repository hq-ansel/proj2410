import contextlib
import os
import sys
import time
import torch
import warnings
from torch.utils.cpp_extension import load

try:
    from torch.cuda import nvtx as _nvtx
except Exception:
    _nvtx = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _extension_build_dir(ext_name: str) -> str:
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    cuda_version = torch.version.cuda
    cuda_tag = "cpu" if not cuda_version else f"cu{cuda_version.replace('.', '')}"
    return os.path.join(root, f"{py_tag}_{cuda_tag}", ext_name)


def _cleanup_stale_extension_lock(ext_name: str) -> None:
    if os.environ.get("EFFICIENTQAT_KEEP_TORCH_EXT_LOCK") == "1":
        return

    build_dir = _extension_build_dir(ext_name)
    lock_path = os.path.join(build_dir, "lock")
    so_path = os.path.join(build_dir, f"{ext_name}.so")
    if not os.path.exists(lock_path) or not os.path.exists(so_path):
        return

    stale_after = max(float(os.environ.get("EFFICIENTQAT_TORCH_EXT_STALE_LOCK_SEC", "300")), 1.0)
    try:
        age = time.time() - os.path.getmtime(lock_path)
    except OSError:
        return
    if age < stale_after:
        return

    try:
        os.remove(lock_path)
    except FileNotFoundError:
        return
    except OSError:
        return

    warnings.warn(
        f"Removed stale torch extension lock: {lock_path}",
        RuntimeWarning,
        stacklevel=2,
    )

def fake_quant_backward(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    qmin: int,
    qmax: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute gradients for fake quantization (stateless backward).

    This is a pure function that computes gradients w.r.t. weight, scale, and
    zero_point given the gradient w.r.t. the quantized output. It directly calls
    the CUDA backward kernel without going through PyTorch's autograd system.

    **Use case**: Custom autograd.Function implementations that need to compute
    quantization gradients manually (e.g., when integrating with frameworks like
    Megatron-LM that have custom gradient accumulation semantics).

    **Contract**:
    - Stateless: no side effects, no .grad assignment
    - Pure function: same inputs → same outputs
    - Only does math backward: no optimizer/DDP/framework logic

    Args:
        grad_output: Gradient w.r.t. quantized weight [out_features, in_features].
                     Must be contiguous CUDA tensor.
        weight: Original (unquantized) weight tensor [out_features, in_features].
                Must be contiguous CUDA tensor.
        scale: Quantization scale parameter [N_groups].
               Must be contiguous CUDA tensor.
        zero_point: Quantization zero point [N_groups] or None.
                    If provided, must be contiguous CUDA tensor.
        qmin: Minimum quantized value (typically 0 for unsigned).
        qmax: Maximum quantized value (typically 2^n_bits - 1).
        group_size: Quantization group size (must be 64, 128, or 256).

    Returns:
        tuple of (grad_weight, grad_scale, grad_zero_point):
        - grad_weight: Gradient w.r.t. original weight [out_features, in_features]
        - grad_scale: Gradient w.r.t. scale [N_groups]
        - grad_zero_point: Gradient w.r.t. zero_point [N_groups] or None

    Raises:
        AssertionError: If input tensors are not contiguous CUDA tensors or
                       if group_size is not in {64, 128, 256}.

    Example:
        >>> weight = torch.randn(1024, 512, device='cuda')
        >>> scale = torch.randn(1024 * 512 // 64, device='cuda')
        >>> zp = torch.randn_like(scale)
        >>> grad_out = torch.randn_like(weight)
        >>> grad_w, grad_s, grad_zp = fake_quant_backward(
        ...     grad_out, weight, scale, zp, 0, 15, 64
        ... )
    """
    assert weight.is_cuda and weight.is_contiguous(), "weight must be contiguous CUDA tensor"
    assert scale.is_cuda and scale.is_contiguous(), "scale must be contiguous CUDA tensor"
    assert grad_output.is_cuda and grad_output.is_contiguous(), "grad_output must be contiguous CUDA tensor"
    assert group_size in (64, 128, 256), f"group_size must be 64/128/256, got {group_size}"

    # Prepare zero_point buffer
    zp = None
    if zero_point is not None:
        assert zero_point.is_cuda and zero_point.is_contiguous(), "zero_point must be contiguous CUDA tensor"
        zp = zero_point

    # Call CUDA backward kernel
    with _nvtx_range("fake_quant_bwd_stateless"):
        grad_weight, grad_scale, grad_zp = _ext.bwd(
            weight, grad_output, scale, zp, int(qmin), int(qmax), int(group_size)
        )

    # Return None for grad_zp if zero_point was None
    grad_zero_point = None if zero_point is None else grad_zp

    return grad_weight, grad_scale, grad_zero_point


def _fake_quant_cuda_cflags() -> list[str]:
    """
    Build NVCC flags for heterogeneous GPUs.
    Override with EFFICIENTQAT_FAKE_QUANT_GENCODE, e.g.:
      -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_120,code=sm_120
    """
    base = ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"]
    custom = os.environ.get("EFFICIENTQAT_FAKE_QUANT_GENCODE", "").strip()
    if custom:
        return base + custom.split()
    # Default covers this repo's common mixed setup: RTX A6000 (sm_86) + RTX 5090 (sm_120).
    return base + [
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_120,code=sm_120",
    ]

_cleanup_stale_extension_lock("fake_quant_ext")

_ext = load(
    name="fake_quant_ext",
    sources=[os.path.join(_THIS_DIR, "fake_quant_ext.cu")],
    extra_cuda_cflags=_fake_quant_cuda_cflags(),
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



def fake_quant_backward(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    qmin: int,
    qmax: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute gradients for fake quantization (stateless backward).

    This is a pure function that computes gradients w.r.t. weight, scale, and
    zero_point given the gradient w.r.t. the quantized output. It directly calls
    the CUDA backward kernel without going through PyTorch's autograd system.

    **Use case**: Custom autograd.Function implementations that need to compute
    quantization gradients manually (e.g., when integrating with frameworks like
    Megatron-LM that have custom gradient accumulation semantics).

    **Contract**:
    - Stateless: no side effects, no .grad assignment
    - Pure function: same inputs → same outputs
    - Only does math backward: no optimizer/DDP/framework logic

    Args:
        grad_output: Gradient w.r.t. quantized weight [out_features, in_features].
                     Must be contiguous CUDA tensor.
        weight: Original (unquantized) weight tensor [out_features, in_features].
                Must be contiguous CUDA tensor.
        scale: Quantization scale parameter [N_groups].
               Must be contiguous CUDA tensor.
        zero_point: Quantization zero point [N_groups] or None.
                    If provided, must be contiguous CUDA tensor.
        qmin: Minimum quantized value (typically 0 for unsigned).
        qmax: Maximum quantized value (typically 2^n_bits - 1).
        group_size: Quantization group size (must be 64, 128, or 256).

    Returns:
        tuple of (grad_weight, grad_scale, grad_zero_point):
        - grad_weight: Gradient w.r.t. original weight [out_features, in_features]
        - grad_scale: Gradient w.r.t. scale [N_groups]
        - grad_zero_point: Gradient w.r.t. zero_point [N_groups] or None

    Raises:
        AssertionError: If input tensors are not contiguous CUDA tensors or
                       if group_size is not in {64, 128, 256}.

    Example:
        >>> weight = torch.randn(1024, 512, device='cuda')
        >>> scale = torch.randn(1024 * 512 // 64, device='cuda')
        >>> zp = torch.randn_like(scale)
        >>> grad_out = torch.randn_like(weight)
        >>> grad_w, grad_s, grad_zp = fake_quant_backward(
        ...     grad_out, weight, scale, zp, 0, 15, 64
        ... )
    """
    assert weight.is_cuda and weight.is_contiguous(), "weight must be contiguous CUDA tensor"
    assert scale.is_cuda and scale.is_contiguous(), "scale must be contiguous CUDA tensor"
    assert grad_output.is_cuda and grad_output.is_contiguous(), "grad_output must be contiguous CUDA tensor"
    assert group_size in (64, 128, 256), f"group_size must be 64/128/256, got {group_size}"

    # Prepare zero_point buffer
    zp = None
    if zero_point is not None:
        assert zero_point.is_cuda and zero_point.is_contiguous(), "zero_point must be contiguous CUDA tensor"
        zp = zero_point

    # Call CUDA backward kernel
    with _nvtx_range("fake_quant_bwd_stateless"):
        grad_weight, grad_scale, grad_zp = _ext.bwd(
            weight, grad_output, scale, zp, int(qmin), int(qmax), int(group_size)
        )

    # Return None for grad_zp if zero_point was None
    grad_zero_point = None if zero_point is None else grad_zp

    return grad_weight, grad_scale, grad_zero_point



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
def fake_quant_backward_seq2bit(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for Seq2Bit fake quantization (stateless backward).

    This is a pure function that computes gradients w.r.t. weight and alpha
    given the gradient w.r.t. the quantized output. It directly calls
    the CUDA backward kernel without going through PyTorch's autograd system.

    **Use case**: Custom autograd.Function implementations that need to compute
    Seq2Bit quantization gradients manually (e.g., when integrating with frameworks
    like Megatron-LM that have custom gradient accumulation semantics).

    **Contract**:
    - Stateless: no side effects, no .grad assignment
    - Pure function: same inputs → same outputs
    - Only does math backward: no optimizer/DDP/framework logic

    Args:
        grad_output: Gradient w.r.t. quantized weight [out_features, in_features].
                     Must be contiguous CUDA tensor.
        weight: Original (unquantized) weight tensor [out_features, in_features].
                Must be contiguous CUDA tensor.
        alpha: Seq2Bit alpha parameter [N_groups].
               Must be contiguous CUDA tensor.
        group_size: Quantization group size (must be 64, 128, or 256).

    Returns:
        tuple of (grad_weight, grad_alpha):
        - grad_weight: Gradient w.r.t. original weight [out_features, in_features]
        - grad_alpha: Gradient w.r.t. alpha [N_groups]

    Raises:
        AssertionError: If input tensors are not contiguous CUDA tensors or
                       if group_size is not in {64, 128, 256}.
    """
    assert weight.is_cuda and weight.is_contiguous(), "weight must be contiguous CUDA tensor"
    assert alpha.is_cuda and alpha.is_contiguous(), "alpha must be contiguous CUDA tensor"
    assert grad_output.is_cuda and grad_output.is_contiguous(), "grad_output must be contiguous CUDA tensor"
    assert group_size in (64, 128, 256), f"group_size must be 64/128/256, got {group_size}"

    # Call CUDA backward kernel
    with _nvtx_range("fake_quant_seq2bit_bwd_stateless"):
        grad_weight, grad_alpha = _ext.fake_quant_ste_seq2bit_bwd_cuda(
            weight, grad_output, alpha, int(group_size)
        )

    return grad_weight, grad_alpha
