import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load_symbol(module_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"test_dynamic_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


clamp_ste = _load_symbol(ROOT / "EfficientQAT/core/quantizer/ops.py", "clamp_ste")
round_ste = _load_symbol(ROOT / "EfficientQAT/core/quantizer/ops.py", "round_ste")


def _seq2bit_ref_fake_quant(x: torch.Tensor, alpha: torch.Tensor, group_size: int) -> torch.Tensor:
    ori_shape = x.shape
    xg = x.reshape(-1, group_size)
    s = clamp_ste(alpha.abs(), 1e-6, 1e4).reshape(-1, 1)
    xn = (xg / s).clamp(-1.0, 1.0)
    q = round_ste((xn + 0.75) / 0.5).clamp(0, 3)
    y = (q * 0.5 - 0.75) * s
    return y.reshape(ori_shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for seq2bit kernel test")
def test_seq2bit_kernel_matches_reference_forward_and_backward():
    try:
        fake_quant_ste_seq2bit = _load_symbol(
            ROOT / "EfficientQAT/core/quantizer/kernel/fake_quant.py",
            "fake_quant_ste_seq2bit",
        )
    except Exception as exc:
        pytest.skip(f"Failed to load fake_quant extension: {exc}")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32
    group_size = 128
    out_features, in_features = 32, 1024

    x0 = torch.randn(out_features, in_features, device=device, dtype=dtype)
    alpha0 = torch.rand(out_features * (in_features // group_size), 1, device=device, dtype=dtype).clamp_min(1e-3)
    grad_out = torch.randn_like(x0)

    x_ref = x0.detach().clone().requires_grad_(True).contiguous()
    a_ref = alpha0.detach().clone().requires_grad_(True).contiguous()
    y_ref = _seq2bit_ref_fake_quant(x_ref, a_ref, group_size)
    (y_ref * grad_out).sum().backward()

    x_ker = x0.detach().clone().requires_grad_(True).contiguous()
    a_ker = alpha0.detach().clone().requires_grad_(True).contiguous()
    y_ker = fake_quant_ste_seq2bit(x_ker, a_ker.reshape(-1).contiguous(), group_size)
    (y_ker * grad_out).sum().backward()

    assert torch.allclose(y_ref, y_ker, atol=1e-6, rtol=1e-5)
    assert torch.allclose(x_ref.grad, x_ker.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(a_ref.grad.reshape(-1), a_ker.grad.reshape(-1), atol=1e-6, rtol=1e-5)
