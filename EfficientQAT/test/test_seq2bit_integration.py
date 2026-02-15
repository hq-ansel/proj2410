import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from EfficientQAT.core.linear.int_quant_linear import (  # noqa: E402
    IntQuantLinear,
    reinit_quant_params,
    sanitize_quant_params,
    set_quant_state,
)
from EfficientQAT.core.quantizer.config import QuantConfig  # noqa: E402
from VeOmni.tasks.quantize.export_tritonv2_quant import (  # noqa: E402
    export_tritonv2_quantized_checkpoint,
)


def test_seq2bit_linear_forward_and_reinit():
    cfg = QuantConfig(quant_type="seq2bit", n_bits=2, group_size=8)
    layer = IntQuantLinear(32, 32, bias=True, prefix="test.linear", config=cfg)
    set_quant_state(layer, weight_quant=True)

    x = torch.randn(2, 32)
    y = layer(x)
    assert y.shape == (2, 32)

    reinit_quant_params(layer)
    repaired = sanitize_quant_params(layer)
    assert repaired == 0

    q = layer.weight_quantizer
    assert q is not None
    assert q.__class__.__name__ == "Seq2BitQuantizer"
    assert q.scale.shape[0] == (32 * 32) // 8
    assert q.zero_point is not None
    q_param_names = dict(q.named_parameters()).keys()
    assert "alpha" in q_param_names
    assert "scale" not in q_param_names


def test_seq2bit_export_torch_pack_path():
    safetensors = pytest.importorskip("safetensors.torch")
    save_file = safetensors.save_file
    load_file = safetensors.load_file

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        state = {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(32, 32, dtype=torch.float16),
            "model.layers.0.mlp.gate_proj.bias": torch.randn(32, dtype=torch.float16),
            "model.layers.0.mlp.gate_proj.weight_quantizer.alpha": torch.rand(128, 1, dtype=torch.float32).clamp_min(
                1e-3
            ),
            "model.layers.0.mlp.gate_proj.weight_quantizer._zero_point": torch.zeros(128, 1, dtype=torch.float32),
        }
        save_file(state, os.path.join(src, "model.safetensors"))

        summary = export_tritonv2_quantized_checkpoint(
            src=src,
            dst=dst,
            save_dequant=True,
            bits=2,
            group_size=8,
            pack_dtype="int32",
            weight_dtype="float16",
        )
        assert summary["quant_type"] == "mixed"
        assert summary["converted_modules"] == ["model.layers.0.mlp.gate_proj"]
        impl = summary.get("quant_impl_by_module", {})
        assert impl.get("model.layers.0.mlp.gate_proj") == "quant_sim_linear"

        with open(os.path.join(dst, "quantize_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["quant_type"] == "mixed"
        assert cfg["group_size"] == 8
        assert cfg["bits"] == 2

        out_state = load_file(os.path.join(dst, "model.safetensors"), device="cpu")
        assert "model.layers.0.mlp.gate_proj.qweight" in out_state
        assert "model.layers.0.mlp.gate_proj.scales" in out_state
        assert "model.layers.0.mlp.gate_proj.g_idx" in out_state
