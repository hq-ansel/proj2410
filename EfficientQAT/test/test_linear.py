import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _load_symbol(module_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"test_dynamic_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


QuantSimLinear = _load_symbol(ROOT / "EfficientQAT/core/linear/quant_sim_linear.py", "QuantSimLinear")
LinearQatParams = _load_symbol(ROOT / "VeOmni/tasks/quantize/export_tritonv2_quant.py", "LinearQatParams")
_pack_one_linear_seq2bit_torch = _load_symbol(
    ROOT / "VeOmni/tasks/quantize/export_tritonv2_quant.py", "_pack_one_linear_seq2bit_torch"
)


def test_seq2bit_dequantize_weight_is_inverse_of_pack():
    out_features, in_features, group_size = 7, 32, 8
    n_groups = in_features // group_size

    torch.manual_seed(0)
    codes = torch.randint(0, 4, (out_features, in_features), dtype=torch.int32)
    levels = codes.to(torch.float32) * 0.5 - 0.75
    alpha_out_g = torch.rand(out_features, n_groups, dtype=torch.float32).clamp_min(1e-3)
    weight = (levels.view(out_features, n_groups, group_size) * alpha_out_g.unsqueeze(-1)).reshape(
        out_features, in_features
    )

    qat = LinearQatParams(
        weight=weight,
        bias=None,
        scale=alpha_out_g.reshape(-1, 1),
        zero_point=torch.zeros(out_features * n_groups, 1, dtype=torch.float32),
    )
    packed, dequant_from_pack = _pack_one_linear_seq2bit_torch(
        prefix="test.linear",
        qat=qat,
        group_size=group_size,
        weight_dtype=torch.float16,
    )

    qsim = QuantSimLinear(
        in_features=in_features,
        out_features=out_features,
        bits=2,
        group_size=group_size,
        impl="seq2bit",
        bias=False,
    )
    qsim.qweight = packed["test.linear.qweight"]
    qsim.scales = packed["test.linear.scales"]
    qsim.g_idx = packed["test.linear.g_idx"]
    dequant_from_qsim = qsim.dequantize_weight(dtype=torch.float16)

    assert torch.equal(packed["test.linear.qweight"].to(torch.int32), codes)
    assert torch.equal(dequant_from_qsim, dequant_from_pack)
