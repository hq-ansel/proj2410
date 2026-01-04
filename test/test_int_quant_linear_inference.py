import torch

from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear
from EfficientQAT.core.quantizer.config import QuantConfig


def _run_inference_error(n_bits: int, group_size: int, seed: int = 123) -> tuple[float, float]:
    torch.manual_seed(seed)
    in_features = 256
    out_features = 128
    batch = 16

    fp_linear = torch.nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        fp_linear.weight.mul_(0.1)
        fp_linear.bias.mul_(0.1)

    config = QuantConfig(
        quant_type="uniform_affine",
        n_bits=n_bits,
        group_size=group_size,
        clamp_method="STE",
        round_method="ste",
    )
    q_linear = IntQuantLinear.from_float("test", fp_linear, config)
    q_linear.use_weight_quant = True
    q_linear.eval()

    x = torch.randn(batch, in_features)
    with torch.no_grad():
        y_fp = fp_linear(x)
        y_q = q_linear(x)

    diff = y_q - y_fp
    rel = diff.norm() / (y_fp.norm() + 1e-12)
    max_abs = diff.abs().max()
    return rel.item(), max_abs.item()


def test_int8_inference_error():
    rel_err, max_abs = _run_inference_error(n_bits=8, group_size=128)
    assert rel_err < 0.02, f"int8 relative error too large: {rel_err:.6f}"
    assert max_abs < 0.1, f"int8 max abs error too large: {max_abs:.6f}"


if __name__ == "__main__":
    rel_err, max_abs = _run_inference_error(n_bits=8, group_size=128)
    print(f"int8 relative error: {rel_err:.6f}")
    print(f"int8 max abs error: {max_abs:.6f}")
