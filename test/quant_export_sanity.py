#!/usr/bin/env python3
"""
Sanity checks for QAT export and quantized inference.

Main diagnostic modes:
- Weight sanity: compare QAT fake-quant weights in `hf_ckpt` vs packed+dequant
  weights in `out`/`out_fixed` (single layer or all layers).
- QAT logits: compare the same model with quant off/on via `set_quant_state`
  to quantify the QAT shift itself.
- Triton vs QAT: compare Triton quant logits to QAT fake-quant logits.

Notes:
- Packed weights are stored transposed; comparisons try both orientations.
- Top-k overlap on one prompt is noisy; prefer diff stats (mean/p99).

Example (QAT fake-quant vs fp16 logits only):
  python /home/ubuntu/data/exp/proj2410/test/quant_export_sanity.py \
    --fp16-dir /home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-int2-preexp/checkpoints/global_step_614/hf_ckpt \
    --quant-dir /home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-int2-preexp/checkpoints/out \
    --qat-compare --qat-only --qat-bits 2 --qat-group-size 128 --qat-quant-type uniform_affine
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
QUANT_TASKS_PATH = REPO_ROOT / "VeOmni" / "tasks" / "quantize"
if str(QUANT_TASKS_PATH) not in sys.path:
    sys.path.append(str(QUANT_TASKS_PATH))

import load_tritonv2_quant  # noqa: E402
from utils.twin_forward_compare import (  # noqa: E402
    TwinCompareConfig,
    compare_models,
    find_irreversible_breakpoint,
    print_compare_report,
)


DEFAULT_FP16_DIR = (
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/"
    "w4g128-int4-dryrun/checkpoints/global_step_1/hf_ckpt"
)
DEFAULT_QUANT_DIR = (
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/"
    "w4g128-int4-dryrun/checkpoints/out"
)


def _device_from_arg(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_fp16(model_dir: str, tokenizer_dir: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _load_quant(model_dir: str, device: str, dtype: str):
    model, tokenizer = load_tritonv2_quant.load_tritonv2_quantized_model(
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        local_files_only=True,
        use_device_map=False,
    )
    model.eval()
    return model, tokenizer


def _should_skip_module(module_name: str, skip_names: set[str]) -> bool:
    return any(module_name == skip or module_name.endswith(f".{skip}") for skip in skip_names)


def _convert_linear_with_skip(module: torch.nn.Module, prefix: str, config, skip_names: set[str]) -> None:
    from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear

    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if _should_skip_module(child_prefix, skip_names):
            continue
        if isinstance(child, torch.nn.Linear) and not isinstance(child, IntQuantLinear):
            setattr(module, name, IntQuantLinear.from_float(child_prefix, child, config))
        else:
            _convert_linear_with_skip(child, child_prefix, config, skip_names)


def _iter_state_dict_files(model_dir: str) -> list[str]:
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        return [os.path.join(model_dir, name) for name in shard_names]

    raise FileNotFoundError(
        f"Cannot find `model.safetensors` or `model.safetensors.index.json` under: {model_dir}"
    )


def _load_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for path in _iter_state_dict_files(model_dir):
        state.update(load_file(path, device="cpu"))
    return state


def _infer_group_size(in_features: int, out_features: int, scale_numel: int) -> int:
    total = in_features * out_features
    if scale_numel <= 0 or total % scale_numel != 0:
        raise ValueError(
            f"Cannot infer group_size: in={in_features}, out={out_features}, scale_numel={scale_numel}"
        )
    return total // scale_numel


def _infer_group_size_from_ckpt(model_dir: str) -> int:
    from safetensors import safe_open

    for path in _iter_state_dict_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".weight_quantizer.scale"):
                    prefix = k[: -len(".weight_quantizer.scale")]
                    scale = f.get_tensor(k)
                    weight = _load_tensor(model_dir, f"{prefix}.weight")
                    return _infer_group_size(weight.shape[1], weight.shape[0], scale.numel())
    raise RuntimeError("Cannot infer group_size: no weight_quantizer.scale tensors found.")


def _load_qat_model(
    model_dir: str,
    device: str,
    dtype: torch.dtype,
    bits: int,
    group_size: int,
    quant_type: str,
    skip_modules: set[str],
):
    from EfficientQAT.core.quantizer.config import QuantConfig as EQuantConfig
    from EfficientQAT.core.linear.int_quant_linear import set_quant_state

    config = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    ).config
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype, attn_implementation="flash_attention_2")
    qcfg = EQuantConfig(quant_type=quant_type, n_bits=bits, group_size=group_size)
    _convert_linear_with_skip(model, prefix="", config=qcfg, skip_names=skip_modules)

    state = _load_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"qat load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"unexpected (first 10): {unexpected[:10]}")

    model.to(device)
    model.eval()
    set_quant_state(model, weight_quant=False)
    return model


def _logits_stats(name: str, logits: torch.Tensor) -> None:
    logits = logits.float()
    nan_count = torch.isnan(logits).sum().item()
    inf_count = torch.isinf(logits).sum().item()
    print(
        f"{name} logits: shape={tuple(logits.shape)} min={logits.min().item():.4g} "
        f"max={logits.max().item():.4g} mean={logits.mean().item():.4g} "
        f"std={logits.std().item():.4g} nan={nan_count} inf={inf_count}"
    )


def _compare_logits(fp16_logits: torch.Tensor, quant_logits: torch.Tensor, topk: int) -> None:
    if fp16_logits.shape != quant_logits.shape:
        print(f"shape mismatch: fp16={tuple(fp16_logits.shape)} quant={tuple(quant_logits.shape)}")
        return
    diff = (fp16_logits.float() - quant_logits.float()).abs()
    print(
        f"abs diff: mean={diff.mean().item():.4g} max={diff.max().item():.4g} "
        f"p99={torch.quantile(diff, 0.99).item():.4g}"
    )
    fp16_topk = torch.topk(fp16_logits, k=topk, dim=-1).indices
    quant_topk = torch.topk(quant_logits, k=topk, dim=-1).indices
    overlap = (fp16_topk == quant_topk).sum().item()
    total = fp16_topk.numel()
    print(f"top-{topk} index overlap: {overlap}/{total} ({overlap / max(1, total):.2%})")


def _load_quantize_config(quant_dir: str) -> dict:
    cfg_path = os.path.join(quant_dir, "quantize_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing quantize_config.json under {quant_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_pack_dtype(pack_dtype: str) -> torch.dtype:
    if pack_dtype == "int32":
        return torch.int32
    if pack_dtype == "int16":
        return torch.int16
    if pack_dtype == "int8":
        return torch.int8
    raise ValueError(f"Unsupported pack_dtype {pack_dtype}")


def _infer_bits_group_from_path(path: str) -> tuple[int | None, int | None]:
    match = re.search(r"w(\d+)g(\d+)", path)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _resolve_safetensor_path(model_dir: str, key: str) -> str:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        filename = weight_map.get(key)
        if filename is None:
            raise KeyError(f"{key} not found in safetensors index under {model_dir}")
        return os.path.join(model_dir, filename)
    return os.path.join(model_dir, "model.safetensors")


def _load_tensor(model_dir: str, key: str) -> torch.Tensor:
    from safetensors import safe_open

    path = _resolve_safetensor_path(model_dir, key)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing safetensors file {path}")
    with safe_open(path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"{key} not found in {path}")
        return f.get_tensor(key)


def _reshape_qat_scale_zp_for_pack(
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size == -1 or group_size > in_features:
        group_size = in_features
    n_groups_in = in_features // group_size
    expected = out_features * n_groups_in
    scale_flat = scale.reshape(-1)
    zero_flat = zero_point.reshape(-1)
    if scale_flat.numel() != expected or zero_flat.numel() != expected:
        raise ValueError(
            f"Unexpected quant param numel: scale={scale_flat.numel()} zero={zero_flat.numel()} expected={expected}"
        )
    scales_out_g = scale_flat.view(out_features, n_groups_in)
    zeros_out_g = zero_flat.view(out_features, n_groups_in)
    return scales_out_g, zeros_out_g


def _resolve_pack_params(args) -> tuple[int, int, str]:
    cfg = None
    if os.path.isfile(os.path.join(args.quant_dir, "quantize_config.json")):
        try:
            cfg = _load_quantize_config(args.quant_dir)
        except Exception:
            cfg = None

    inferred_bits, inferred_group = _infer_bits_group_from_path(args.fp16_dir)
    if inferred_bits is None and inferred_group is None:
        inferred_bits, inferred_group = _infer_bits_group_from_path(args.quant_dir)

    bits = args.pack_bits
    if bits is None and cfg is not None and cfg.get("bits") is not None and "--pack-bits" not in sys.argv:
        bits = int(cfg["bits"])
    if bits is None:
        if "--qat-bits" in sys.argv:
            bits = int(args.qat_bits)
        elif inferred_bits is not None:
            bits = inferred_bits
        else:
            bits = int(args.qat_bits)

    group_size = args.pack_group_size
    if (
        group_size is None
        and cfg is not None
        and cfg.get("group_size") is not None
        and "--pack-group-size" not in sys.argv
    ):
        group_size = int(cfg["group_size"])
    if group_size is None:
        if "--qat-group-size" in sys.argv and args.qat_group_size is not None:
            group_size = int(args.qat_group_size)
        elif inferred_group is not None:
            group_size = inferred_group
        else:
            group_size = _infer_group_size_from_ckpt(args.fp16_dir)

    return int(bits), int(group_size), args.pack_dtype


def _clamp_qparams(scale: torch.Tensor, zero_point: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    min_scale = 1e-5
    max_scale = 1e4
    sign = torch.where(scale >= 0, torch.ones_like(scale), -torch.ones_like(scale))
    scale = torch.clamp(scale.abs(), min_scale, max_scale).to(dtype=scale.dtype) * sign
    maxq = (1 << bits) - 1
    zero_point = torch.clamp(torch.round(zero_point), 0, maxq).to(dtype=zero_point.dtype)
    return scale, zero_point


def _fake_quant_weight(weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int, group_size: int):
    maxq = (1 << bits) - 1
    w = weight.float()
    s = scale.reshape(-1, 1).float()
    z = zero_point.reshape(-1, 1).float()
    s, z = _clamp_qparams(s, z, bits)
    if group_size is None:
        raise ValueError("group_size must not be None for QAT fake-quant.")
    x = w.reshape(-1, group_size)
    x_int = torch.round(x / s) + z
    x_int = torch.clamp(x_int, 0, maxq)
    x_dequant = (x_int - z) * s
    return x_dequant.reshape(w.shape)


def _dequant_packed_weight(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    group_size: int,
    pack_dtype: str,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear

    pack_dtype_map = {"int32": torch.int32, "int16": torch.int16, "int8": torch.int8}
    if pack_dtype not in pack_dtype_map:
        raise ValueError(f"Unsupported pack_dtype {pack_dtype}")

    qlinear = TritonV2QuantLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=pack_dtype_map[pack_dtype],
    )
    qlinear.post_init()
    # Some code paths expect dequant_dtype to be present.
    if not hasattr(qlinear, "dequant_dtype"):
        qlinear.dequant_dtype = torch.float16
    qlinear.qweight = qweight
    qlinear.qzeros = qzeros
    qlinear.scales = scales
    qlinear.g_idx = g_idx
    return qlinear.dequantize_weight()


def _export_packed_checkpoint(
    fp16_dir: str,
    quant_dir: str,
    bits: int,
    group_size: int,
    pack_dtype: str,
) -> dict:
    import export_tritonv2_quant

    return export_tritonv2_quant.export_tritonv2_quantized_checkpoint(
        src=fp16_dir,
        dst=quant_dir,
        bits=bits,
        group_size=group_size,
        pack_dtype=pack_dtype,
        weight_dtype="auto",
        qat_param_order="out_major",
        exclude=None,
    )


def _compare_int_weight(
    fp16_dir: str,
    bits: int,
    group_size: int,
    pack_dtype: str,
    layer: str | None,
    limit: int | None,
    verbose: bool,
    print_one: bool,
    save_report: bool,
    report_dir: str,
) -> None:
    from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear
    from EfficientQAT.core.quantizer.config import QuantConfig as EQuantConfig
    from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear

    state = _load_state_dict(fp16_dir)
    prefixes = sorted(
        {k[: -len(".weight_quantizer.scale")] for k in state if k.endswith(".weight_quantizer.scale")}
    )
    if layer is not None:
        prefixes = [layer]
    if limit is not None:
        prefixes = prefixes[:limit]

    results = []
    errors = []
    pack_dtype_t = _parse_pack_dtype(pack_dtype)

    for prefix in prefixes:
        try:
            weight = state[f"{prefix}.weight"].float()
            scale = state[f"{prefix}.weight_quantizer.scale"].float()
            zero_point = state[f"{prefix}.weight_quantizer.zero_point"].float()
            bias = state.get(f"{prefix}.bias")
            if bias is not None:
                bias = bias.float()

            out_features, in_features = weight.shape
            resolved_group = group_size
            if resolved_group is None:
                resolved_group = _infer_group_size(in_features, out_features, scale.numel())

            scales_out_g, zeros_out_g = _reshape_qat_scale_zp_for_pack(
                scale,
                zero_point,
                out_features=out_features,
                in_features=in_features,
                group_size=resolved_group,
            )
            scales_out_g, zeros_out_g = _clamp_qparams(scales_out_g, zeros_out_g, bits)

            # PackableQuantLinear path
            linear = torch.nn.Linear(in_features, out_features, bias=bias is not None)
            linear.weight.data.copy_(weight)
            if bias is not None and linear.bias is not None:
                linear.bias.data.copy_(bias)

            qlinear = TritonV2QuantLinear(
                bits=bits,
                group_size=resolved_group,
                desc_act=False,
                sym=False,
                in_features=in_features,
                out_features=out_features,
                bias=bias is not None,
                pack_dtype=pack_dtype_t,
            )
            qlinear.post_init()
            qlinear.debug_int_weight = True
            qlinear.pack(linear=linear, scales=scales_out_g, zeros=zeros_out_g)
            int_pack = qlinear.int_weight_debug
            if int_pack is None:
                raise RuntimeError("PackableQuantLinear did not store int_weight_debug")

            # IntQuantLinear path
            qcfg = EQuantConfig(quant_type="uniform_affine", n_bits=bits, group_size=resolved_group)
            qat_linear = IntQuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias is not None,
                prefix=prefix,
                config=qcfg,
            )
            qat_linear.weight.data.copy_(weight)
            if bias is not None and qat_linear.bias is not None:
                qat_linear.bias.data.copy_(bias)
            qat_linear.weight_quantizer.scale.data.copy_(scale)
            qat_linear.weight_quantizer.zero_point.data.copy_(zero_point)
            qat_linear.debug_int_weight = True
            int_qat = qat_linear.get_int_weight()
            if int_qat.device != int_pack.device:
                int_qat = int_qat.to(int_pack.device)

            if int_qat.shape != int_pack.shape:
                raise ValueError(
                    f"int_weight shape mismatch: qat={tuple(int_qat.shape)} pack={tuple(int_pack.shape)}"
                )

            diff_mask = int_qat != int_pack
            mismatch = int(diff_mask.sum().item())
            total = int(diff_mask.numel())
            abs_diff = (int_qat - int_pack).abs()
            max_abs = int(abs_diff.max().item()) if total else 0
            mean_abs = float(abs_diff.float().mean().item()) if total else 0.0

            results.append(
                {
                    "layer": prefix,
                    "mismatch": mismatch,
                    "total": total,
                    "ratio": mismatch / max(total, 1),
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                }
            )
            if verbose:
                print(
                    f"{prefix}: mismatched={mismatch}/{total} "
                    f"ratio={mismatch / max(total, 1):.4g} max_abs={max_abs} mean_abs={mean_abs:.4g}"
                )
        except Exception as exc:
            errors.append((prefix, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"{prefix}: ERROR {type(exc).__name__}: {exc}")

    if not results:
        print("no int_weight layers compared successfully")
        if errors:
            print(f"errors: {len(errors)} (first 10):")
            for name, err in errors[:10]:
                print(f"{name}: {err}")
        return

    total_mismatch = sum(r["mismatch"] for r in results)
    total_elems = sum(r["total"] for r in results)
    worst = max(results, key=lambda r: r["ratio"])

    print(
        f"int_weight compared: layers={len(results)} errors={len(errors)} "
        f"mismatch={total_mismatch}/{total_elems} "
        f"ratio={total_mismatch / max(total_elems, 1):.4g}"
    )
    print(
        f"worst layer: {worst['layer']} ratio={worst['ratio']:.4g} "
        f"max_abs={worst['max_abs']} mean_abs={worst['mean_abs']:.4g}"
    )

    if print_one:
        try:
            prefix = worst['layer']
            weight = state[f"{prefix}.weight"].float()
            scale = state[f"{prefix}.weight_quantizer.scale"].float()
            zero_point = state[f"{prefix}.weight_quantizer.zero_point"].float()

            out_features, in_features = weight.shape
            resolved_group = group_size
            if resolved_group is None:
                resolved_group = _infer_group_size(in_features, out_features, scale.numel())

            scales_out_g, zeros_out_g = _reshape_qat_scale_zp_for_pack(
                scale, zero_point, out_features=out_features, in_features=in_features, group_size=resolved_group
            )
            scales_out_g, zeros_out_g = _clamp_qparams(scales_out_g, zeros_out_g, bits)

            linear = torch.nn.Linear(in_features, out_features, bias=False)
            linear.weight.data.copy_(weight)

            qlinear = TritonV2QuantLinear(
                bits=bits,
                group_size=resolved_group,
                desc_act=False,
                sym=False,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                pack_dtype=pack_dtype_t,
            )
            qlinear.post_init()
            qlinear.debug_int_weight = True
            qlinear.pack(linear=linear, scales=scales_out_g, zeros=zeros_out_g)
            int_pack = qlinear.int_weight_debug

            qcfg = EQuantConfig(quant_type="uniform_affine", n_bits=bits, group_size=resolved_group)
            qat_linear = IntQuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                prefix=prefix,
                config=qcfg,
            )
            qat_linear.weight.data.copy_(weight)
            qat_linear.weight_quantizer.scale.data.copy_(scale)
            qat_linear.weight_quantizer.zero_point.data.copy_(zero_point)
            qat_linear.debug_int_weight = True
            int_qat = qat_linear.get_int_weight()

            group_len = min(resolved_group, in_features)
            weight_row = weight[0, :group_len].tolist()
            int_pack_row = int_pack[0, :group_len].tolist()
            int_qat_row = int_qat[0, :group_len].tolist()
            scale_val = float(scales_out_g[0, 0].item())
            zero_val = float(zeros_out_g[0, 0].item())
            print(f"debug worst layer: {prefix}")
            print(f"weight[0,:{group_len}] = {weight_row}")
            print(f"scale[0,0] = {scale_val}")
            print(f"zero_point[0,0] = {zero_val}")
            print(f"int_weight_pack[0,:{group_len}] = {int_pack_row}")
            print(f"int_weight_qat[0,:{group_len}] = {int_qat_row}")
        except Exception as e:
            print(f"warning: failed to print worst layer details: {e}")

    if save_report:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "int_weight_report.json")
        payload = {"results": results, "errors": errors}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"saved int_weight report to {report_path}")


def _compare_layer_weights_stats(
    fp16_dir: str,
    quant_dir: str,
    layer: str,
    bits: int,
    group_size: int | None,
    pack_dtype: str,
    weight_dtype: torch.dtype | None,
) -> dict:
    weight = _load_tensor(fp16_dir, f"{layer}.weight")
    scale = _load_tensor(fp16_dir, f"{layer}.weight_quantizer.scale")
    zero_point = _load_tensor(fp16_dir, f"{layer}.weight_quantizer.zero_point")
    if weight_dtype is not None:
        weight = weight.to(dtype=weight_dtype)
        scale = scale.to(dtype=weight_dtype)
        zero_point = zero_point.to(dtype=weight_dtype)
    w_qat = _fake_quant_weight(weight, scale, zero_point, bits=bits, group_size=group_size)

    qweight = _load_tensor(quant_dir, f"{layer}.qweight")
    qzeros = _load_tensor(quant_dir, f"{layer}.qzeros")
    scales = _load_tensor(quant_dir, f"{layer}.scales")
    g_idx = _load_tensor(quant_dir, f"{layer}.g_idx")
    w_packed = _dequant_packed_weight(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        g_idx=g_idx,
        bits=bits,
        group_size=group_size,
        pack_dtype=pack_dtype,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
    )

    w_qat = w_qat.float()
    w_packed = w_packed.float()

    abs_w = w_qat.abs()
    abs_mean = abs_w.mean().item()
    abs_max = abs_w.max().item()
    abs_p99 = torch.quantile(abs_w, 0.99).item()

    def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
        d = (a - b).abs()
        return {
            "mean": d.mean().item(),
            "max": d.max().item(),
            "p99": torch.quantile(d, 0.99).item(),
        }

    diff_direct = None
    diff_transpose = None
    if w_qat.shape == w_packed.shape:
        diff_direct = _diff_stats(w_qat, w_packed)
    if w_packed.T.shape == w_qat.shape:
        diff_transpose = _diff_stats(w_qat, w_packed.T)
    if diff_direct is None and diff_transpose is None:
        raise ValueError(f"shape mismatch: fp16={tuple(w_qat.shape)} packed={tuple(w_packed.shape)}")

    transposed = False
    chosen = diff_direct
    if diff_transpose is not None and (diff_direct is None or diff_transpose["mean"] < diff_direct["mean"]):
        chosen = diff_transpose
        transposed = True
    return {
        "abs_mean": abs_mean,
        "abs_max": abs_max,
        "abs_p99": abs_p99,
        "diff_mean": chosen["mean"],
        "diff_max": chosen["max"],
        "diff_p99": chosen["p99"],
        "transposed": transposed,
        "diff_direct_mean": None if diff_direct is None else diff_direct["mean"],
        "diff_direct_max": None if diff_direct is None else diff_direct["max"],
        "diff_direct_p99": None if diff_direct is None else diff_direct["p99"],
        "diff_transpose_mean": None if diff_transpose is None else diff_transpose["mean"],
        "diff_transpose_max": None if diff_transpose is None else diff_transpose["max"],
        "diff_transpose_p99": None if diff_transpose is None else diff_transpose["p99"],
    }


def _compare_layer_weights(
    fp16_dir: str,
    quant_dir: str,
    layer: str,
    weight_dtype: torch.dtype | None,
) -> None:
    cfg = _load_quantize_config(quant_dir)
    bits = int(cfg.get("bits"))
    group_size = int(cfg.get("group_size")) if cfg.get("group_size") is not None else None
    pack_dtype = cfg.get("pack_dtype", "int32")

    stats = _compare_layer_weights_stats(
        fp16_dir, quant_dir, layer, bits, group_size, pack_dtype, weight_dtype
    )
    print(
        f"layer {layer} weight stats: abs_mean={stats['abs_mean']:.4g} "
        f"abs_max={stats['abs_max']:.4g} p99={stats['abs_p99']:.4g}"
    )
    if stats["diff_direct_mean"] is not None:
        print(
            f"diff direct: mean={stats['diff_direct_mean']:.4g} "
            f"max={stats['diff_direct_max']:.4g} p99={stats['diff_direct_p99']:.4g}"
        )
    if stats["diff_transpose_mean"] is not None:
        print(
            f"diff transpose: mean={stats['diff_transpose_mean']:.4g} "
            f"max={stats['diff_transpose_max']:.4g} p99={stats['diff_transpose_p99']:.4g}"
        )
    print(
        f"diff {'transpose' if stats['transposed'] else 'direct'}: mean={stats['diff_mean']:.4g} "
        f"max={stats['diff_max']:.4g} p99={stats['diff_p99']:.4g}"
    )
    if stats["transposed"]:
        print("note: packed weights shape matches fp16 only after transpose.")


def _find_qat_prefixes_in_ckpt(model_dir: str) -> list[str]:
    from safetensors import safe_open

    prefixes = set()
    for path in _iter_state_dict_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".weight_quantizer.scale"):
                    prefixes.add(k[: -len(".weight_quantizer.scale")])
    return sorted(prefixes)


def _compare_all_layers(
    fp16_dir: str,
    quant_dir: str,
    top_n: int,
    verbose: bool,
    limit: int | None,
    weight_dtype: torch.dtype | None,
) -> None:
    cfg = _load_quantize_config(quant_dir)
    bits = int(cfg.get("bits"))
    group_size = int(cfg.get("group_size")) if cfg.get("group_size") is not None else None
    pack_dtype = cfg.get("pack_dtype", "int32")
    converted = cfg.get("converted_modules") or []
    layers = converted if converted else _find_qat_prefixes_in_ckpt(fp16_dir)
    if limit is not None:
        layers = layers[:limit]

    results = []
    errors = []
    for layer in layers:
        try:
            stats = _compare_layer_weights_stats(
                fp16_dir, quant_dir, layer, bits, group_size, pack_dtype, weight_dtype
            )
            stats["layer"] = layer
            results.append(stats)
            if verbose:
                rel = stats["diff_mean"] / max(stats["abs_mean"], 1e-12)
                direct = stats["diff_direct_mean"]
                transpose = stats["diff_transpose_mean"]
                direct_s = "na" if direct is None else f"{direct:.4g}"
                transpose_s = "na" if transpose is None else f"{transpose:.4g}"
                print(
                    f"{layer}: abs_mean={stats['abs_mean']:.4g} direct={direct_s} "
                    f"transpose={transpose_s} best={stats['diff_mean']:.4g} "
                    f"rel={rel:.4g} transposed={stats['transposed']}"
                )
        except Exception as exc:
            errors.append((layer, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"{layer}: ERROR {type(exc).__name__}: {exc}")

    if not results:
        print("no layers compared successfully")
        if errors:
            print(f"errors: {len(errors)} (first 10):")
            for layer, err in errors[:10]:
                print(f"{layer}: {err}")
        return

    transposed_count = sum(1 for r in results if r["transposed"])
    avg_abs_mean = sum(r["abs_mean"] for r in results) / len(results)
    avg_diff_mean = sum(r["diff_mean"] for r in results) / len(results)
    avg_rel_mean = sum(r["diff_mean"] / max(r["abs_mean"], 1e-12) for r in results) / len(results)

    print(
        f"layers compared: {len(results)} errors: {len(errors)} "
        f"transposed_used: {transposed_count}"
    )
    print(
        f"avg abs_mean={avg_abs_mean:.4g} avg diff_mean={avg_diff_mean:.4g} "
        f"avg rel_mean={avg_rel_mean:.4g}"
    )

    worst_rel = sorted(
        results, key=lambda r: r["diff_mean"] / max(r["abs_mean"], 1e-12), reverse=True
    )[:top_n]
    worst_abs = sorted(results, key=lambda r: r["diff_mean"], reverse=True)[:top_n]

    print("worst rel_mean:")
    for r in worst_rel:
        rel = r["diff_mean"] / max(r["abs_mean"], 1e-12)
        direct = r["diff_direct_mean"]
        transpose = r["diff_transpose_mean"]
        direct_s = "na" if direct is None else f"{direct:.4g}"
        transpose_s = "na" if transpose is None else f"{transpose:.4g}"
        print(
            f"{r['layer']}: rel={rel:.4g} direct={direct_s} transpose={transpose_s} "
            f"best={r['diff_mean']:.4g} abs_mean={r['abs_mean']:.4g} "
            f"transposed={r['transposed']}"
        )

    print("worst diff_mean:")
    for r in worst_abs:
        rel = r["diff_mean"] / max(r["abs_mean"], 1e-12)
        direct = r["diff_direct_mean"]
        transpose = r["diff_transpose_mean"]
        direct_s = "na" if direct is None else f"{direct:.4g}"
        transpose_s = "na" if transpose is None else f"{transpose:.4g}"
        print(
            f"{r['layer']}: best={r['diff_mean']:.4g} rel={rel:.4g} "
            f"direct={direct_s} transpose={transpose_s} "
            f"abs_mean={r['abs_mean']:.4g} transposed={r['transposed']}"
        )


def _dequant_int_quant_weight(module) -> torch.Tensor:
    if module.weight_quantizer is None:
        raise RuntimeError("IntQuantLinear is missing weight_quantizer")
    quantizer = module.weight_quantizer
    target_device = module.weight.device
    if hasattr(quantizer, "scale") and quantizer.scale.device != target_device:
        quantizer.to(target_device)
    if hasattr(quantizer, "quantization_position_ratio"):
        quantizer.quantization_position_ratio = 1.0
        quantizer.group_mask = None
        quantizer.interpolate_ratio = 0.0
    with torch.no_grad():
        return quantizer(module.weight)


def _dequant_triton_weight(module) -> torch.Tensor:
    target_device = module.qweight.device
    if hasattr(module, "qzeros") and module.qzeros.device != target_device:
        module.qzeros = module.qzeros.to(device=target_device)
    if hasattr(module, "scales") and module.scales.device != target_device:
        module.scales = module.scales.to(device=target_device)
    if hasattr(module, "g_idx") and module.g_idx.device != target_device:
        module.g_idx = module.g_idx.to(device=target_device)
    if not hasattr(module, "wf_unsqueeze_zero") or not hasattr(module, "wf_unsqueeze_neg_one"):
        module.post_init()
    if module.wf_unsqueeze_zero.device != target_device:
        module.wf_unsqueeze_zero = module.wf_unsqueeze_zero.to(device=target_device)
    if module.wf_unsqueeze_neg_one.device != target_device:
        module.wf_unsqueeze_neg_one = module.wf_unsqueeze_neg_one.to(device=target_device)
    if not hasattr(module, "dequant_dtype"):
        module.dequant_dtype = module.scales.dtype if hasattr(module, "scales") else torch.float16
    with torch.no_grad():
        return module.dequantize_weight()


def _compare_module_dequant_weights(
    fp16_dir: str,
    quant_dir: str,
    device: str,
    dtype: torch.dtype,
    dtype_name: str,
    bits: int,
    group_size: int | None,
    quant_type: str,
    skip_modules: set[str],
    limit: int | None,
    verbose: bool,
) -> None:
    from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear
    from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear

    if group_size is None:
        group_size = _infer_group_size_from_ckpt(fp16_dir)

    qat_model = _load_qat_model(
        fp16_dir,
        device=device,
        dtype=dtype,
        bits=bits,
        group_size=int(group_size),
        quant_type=quant_type,
        skip_modules=skip_modules,
    )
    quant_model, _ = _load_quant(quant_dir, device, dtype_name)

    qat_modules = {
        name: mod for name, mod in qat_model.named_modules() if isinstance(mod, IntQuantLinear)
    }
    triton_modules = {
        name: mod for name, mod in quant_model.named_modules() if isinstance(mod, TritonV2QuantLinear)
    }

    shared = sorted(set(qat_modules) & set(triton_modules))
    if limit is not None:
        shared = shared[:limit]

    missing_qat = sorted(set(triton_modules) - set(qat_modules))
    missing_triton = sorted(set(qat_modules) - set(triton_modules))

    results = []
    errors = []
    for name in shared:
        qat_mod = qat_modules[name]
        triton_mod = triton_modules[name]
        try:
            w_qat = _dequant_int_quant_weight(qat_mod).float()
            w_triton = _dequant_triton_weight(triton_mod).float()
            if w_qat.device != w_triton.device:
                w_triton = w_triton.to(w_qat.device)

            abs_w = w_qat.abs()
            abs_mean = abs_w.mean().item()
            abs_max = abs_w.max().item()
            abs_p99 = torch.quantile(abs_w, 0.99).item()

            def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
                d = (a - b).abs()
                return {
                    "mean": d.mean().item(),
                    "max": d.max().item(),
                    "p99": torch.quantile(d, 0.99).item(),
                }

            diff_direct = None
            diff_transpose = None
            if w_qat.shape == w_triton.shape:
                diff_direct = _diff_stats(w_qat, w_triton)
            if w_triton.T.shape == w_qat.shape:
                diff_transpose = _diff_stats(w_qat, w_triton.T)
            if diff_direct is None and diff_transpose is None:
                raise ValueError(
                    f"shape mismatch: qat={tuple(w_qat.shape)} triton={tuple(w_triton.shape)}"
                )

            transposed = False
            chosen = diff_direct
            if diff_transpose is not None and (diff_direct is None or diff_transpose["mean"] < diff_direct["mean"]):
                chosen = diff_transpose
                transposed = True

            results.append(
                {
                    "layer": name,
                    "abs_mean": abs_mean,
                    "abs_max": abs_max,
                    "abs_p99": abs_p99,
                    "diff_mean": chosen["mean"],
                    "diff_max": chosen["max"],
                    "diff_p99": chosen["p99"],
                    "transposed": transposed,
                    "diff_direct_mean": None if diff_direct is None else diff_direct["mean"],
                    "diff_direct_max": None if diff_direct is None else diff_direct["max"],
                    "diff_direct_p99": None if diff_direct is None else diff_direct["p99"],
                    "diff_transpose_mean": None if diff_transpose is None else diff_transpose["mean"],
                    "diff_transpose_max": None if diff_transpose is None else diff_transpose["max"],
                    "diff_transpose_p99": None if diff_transpose is None else diff_transpose["p99"],
                }
            )
            if verbose:
                rel = chosen["mean"] / max(abs_mean, 1e-12)
                direct = diff_direct
                transpose = diff_transpose
                direct_s = "na" if direct is None else f"{direct['mean']:.4g}"
                transpose_s = "na" if transpose is None else f"{transpose['mean']:.4g}"
                print(
                    f"{name}: abs_mean={abs_mean:.4g} direct={direct_s} "
                    f"transpose={transpose_s} best={chosen['mean']:.4g} "
                    f"rel={rel:.4g} transposed={transposed}"
                )
        except Exception as exc:
            errors.append((name, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"{name}: ERROR {type(exc).__name__}: {exc}")

    if not results:
        print("no modules compared successfully")
        if errors:
            print(f"errors: {len(errors)} (first 10):")
            for layer, err in errors[:10]:
                print(f"{layer}: {err}")
        return

    transposed_count = sum(1 for r in results if r["transposed"])
    avg_abs_mean = sum(r["abs_mean"] for r in results) / len(results)
    avg_diff_mean = sum(r["diff_mean"] for r in results) / len(results)
    avg_rel_mean = sum(r["diff_mean"] / max(r["abs_mean"], 1e-12) for r in results) / len(results)

    print(
        f"modules compared: {len(results)} errors: {len(errors)} "
        f"transposed_used: {transposed_count}"
    )
    if missing_qat:
        print(f"missing in qat: {len(missing_qat)} (first 10): {missing_qat[:10]}")
    if missing_triton:
        print(f"missing in triton: {len(missing_triton)} (first 10): {missing_triton[:10]}")
    print(
        f"avg abs_mean={avg_abs_mean:.4g} avg diff_mean={avg_diff_mean:.4g} "
        f"avg rel_mean={avg_rel_mean:.4g}"
    )

    worst_rel = sorted(
        results, key=lambda r: r["diff_mean"] / max(r["abs_mean"], 1e-12), reverse=True
    )[:10]
    worst_abs = sorted(results, key=lambda r: r["diff_mean"], reverse=True)[:10]

    print("worst rel_mean:")
    for r in worst_rel:
        rel = r["diff_mean"] / max(r["abs_mean"], 1e-12)
        direct = r["diff_direct_mean"]
        transpose = r["diff_transpose_mean"]
        direct_s = "na" if direct is None else f"{direct:.4g}"
        transpose_s = "na" if transpose is None else f"{transpose:.4g}"
        print(
            f"{r['layer']}: rel={rel:.4g} direct={direct_s} transpose={transpose_s} "
            f"best={r['diff_mean']:.4g} abs_mean={r['abs_mean']:.4g} "
            f"transposed={r['transposed']}"
        )

    print("worst diff_mean:")
    for r in worst_abs:
        rel = r["diff_mean"] / max(r["abs_mean"], 1e-12)
        direct = r["diff_direct_mean"]
        transpose = r["diff_transpose_mean"]
        direct_s = "na" if direct is None else f"{direct:.4g}"
        transpose_s = "na" if transpose is None else f"{transpose:.4g}"
        print(
            f"{r['layer']}: best={r['diff_mean']:.4g} rel={rel:.4g} "
            f"direct={direct_s} transpose={transpose_s} "
            f"abs_mean={r['abs_mean']:.4g} transposed={r['transposed']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check exported tritonv2 quant checkpoint vs fp16.")
    parser.add_argument(
        "--fp16-dir",
        default=DEFAULT_FP16_DIR,
        help="HF checkpoint dir for fp16 model (hf_ckpt).",
    )
    parser.add_argument(
        "--quant-dir",
        default=DEFAULT_QUANT_DIR,
        help="Exported tritonv2 quant dir (out).",
    )
    parser.add_argument("--tokenizer-dir", default=None, help="Tokenizer dir; defaults to --fp16-dir.")
    parser.add_argument("--prompt", default="Hello world!", help="Prompt used for a single forward pass.")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--weight-dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Override dtype for QAT reference weights in weight comparisons; defaults to --dtype.",
    )
    parser.add_argument(
        "--export-pack",
        action="store_true",
        help="Pack QAT checkpoint into tritonv2 quant format under --quant-dir.",
    )
    parser.add_argument("--pack-bits", type=int, default=None, help="Pack bits override for --export-pack.")
    parser.add_argument(
        "--pack-group-size",
        type=int,
        default=None,
        help="Pack group size override for --export-pack; inferred if omitted.",
    )
    parser.add_argument("--pack-dtype", default="int32", choices=["int32", "int16", "int8"])
    parser.add_argument(
        "--compare-int-weight",
        action="store_true",
        help="Compare int_weight between IntQuantLinear and PackableQuantLinear.",
    )
    parser.add_argument("--compare-int-layer", default=None, help="Layer prefix for int_weight comparison.")
    parser.add_argument("--compare-int-limit", type=int, default=None, help="Limit int_weight layers.")
    parser.add_argument("--compare-int-verbose", action="store_true", help="Verbose int_weight comparison.")
    parser.add_argument(
        "--compare-int-print-one",
        action="store_true",
        help="Print one layer's weight/scale/zero/int_weight slice for debugging.",
    )
    parser.add_argument(
        "--save-int-weight-report",
        action="store_true",
        help="Save int_weight diff report under --quant-dir.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-k index overlap to report.")
    parser.add_argument("--compare-layer", default=None, help="Module prefix to compare packed vs QAT fake-quant weights.")
    parser.add_argument("--skip-logits", action="store_true", help="Skip logits check; only compare weights.")
    parser.add_argument("--qat-compare", action="store_true", help="Compare QAT fake-quant logits vs fp16 logits.")
    parser.add_argument("--qat-bits", type=int, default=8, help="QAT bits (used for fake-quant).")
    parser.add_argument("--qat-group-size", type=int, default=None, help="QAT group size; inferred if omitted.")
    parser.add_argument("--qat-quant-type", default="uniform_affine", choices=["uniform_affine", "gradual"])
    parser.add_argument("--qat-skip-modules", default="lm_head", help="Comma-separated module names to skip quant.")
    parser.add_argument("--qat-only", action="store_true", help="Only run QAT comparison and exit.")
    parser.add_argument(
        "--compare-qattq",
        action="store_true",
        help="Compare QAT fake-quant logits vs Triton quant logits.",
    )
    parser.add_argument("--compare-all-layers", action="store_true", help="Compare all quantized layers.")
    parser.add_argument("--compare-all-top", type=int, default=10, help="Top N layers to show in summary.")
    parser.add_argument("--compare-all-verbose", action="store_true", help="Print per-layer stats.")
    parser.add_argument("--compare-all-limit", type=int, default=None, help="Limit number of layers to compare.")
    parser.add_argument(
        "--compare-module-weights",
        action="store_true",
        help="Compare IntQuantLinear vs TritonV2QuantLinear dequant weights in model.",
    )
    parser.add_argument(
        "--compare-module-verbose",
        action="store_true",
        help="Print per-module stats for --compare-module-weights.",
    )
    parser.add_argument(
        "--compare-module-limit",
        type=int,
        default=None,
        help="Limit number of modules to compare for --compare-module-weights.",
    )
    parser.add_argument(
        "--twin-compare",
        action="store_true",
        help="Compare module outputs between fp16 and quant models.",
    )
    parser.add_argument(
        "--twin-breakpoint",
        action="store_true",
        help="Locate irreversible breakpoint via prefix injection.",
    )
    parser.add_argument(
        "--twin-mode",
        default="decoupled",
        choices=["decoupled", "coupled"],
        help="Twin compare mode for module-level comparison.",
    )
    parser.add_argument(
        "--twin-topk",
        type=int,
        default=10,
        help="Top K mismatched modules to print.",
    )
    parser.add_argument(
        "--twin-metric",
        default="rel_l2",
        choices=["rel_l2", "max_abs", "mean_abs", "cos"],
        help="Metric used to rank mismatched modules.",
    )
    parser.add_argument("--twin-include-regex", default=None, help="Regex to include module names.")
    parser.add_argument("--twin-exclude-regex", default=None, help="Regex to exclude module names.")
    parser.add_argument("--twin-dump-dir", default=None, help="Dump twin compare reports to directory.")
    parser.add_argument(
        "--twin-breakpoint-strategy",
        default="bisect",
        choices=["bisect", "linear"],
        help="Search strategy for breakpoint localization.",
    )
    parser.add_argument(
        "--twin-breakpoint-threshold",
        type=float,
        default=0.02,
        help="Logits rel_l2 threshold for breakpoint recovery.",
    )
    args = parser.parse_args()

    pack_bits = None
    pack_group_size = None
    pack_dtype = None
    if args.export_pack or args.compare_int_weight:
        pack_bits, pack_group_size, pack_dtype = _resolve_pack_params(args)
        if args.export_pack:
            summary = _export_packed_checkpoint(
                fp16_dir=args.fp16_dir,
                quant_dir=args.quant_dir,
                bits=pack_bits,
                group_size=pack_group_size,
                pack_dtype=pack_dtype,
            )
            print(
                f"exported packed checkpoint: bits={summary.get('bits')} "
                f"group_size={summary.get('group_size')} pack_dtype={summary.get('pack_dtype')}"
            )
        if args.compare_int_weight:
            _compare_int_weight(
                fp16_dir=args.fp16_dir,
                bits=pack_bits,
                group_size=pack_group_size,
                pack_dtype=pack_dtype,
                layer=args.compare_int_layer,
                limit=args.compare_int_limit,
                verbose=args.compare_int_verbose,
                print_one=args.compare_int_print_one,
                save_report=args.save_int_weight_report,
                report_dir=args.quant_dir,
            )
            if args.skip_logits:
                remaining = any(
                    [
                        args.compare_layer,
                        args.compare_all_layers,
                        args.compare_module_weights,
                        args.qat_compare,
                        args.compare_qattq,
                        args.twin_compare,
                        args.twin_breakpoint,
                    ]
                )
                if not remaining:
                    return

    # 1) Weight-only checks (single layer or all layers).
    weight_dtype = None
    if args.weight_dtype is not None:
        weight_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.weight_dtype]

    if args.compare_layer:
        _compare_layer_weights(args.fp16_dir, args.quant_dir, args.compare_layer, weight_dtype)
        if args.skip_logits and not (args.twin_compare or args.twin_breakpoint):
            return

    if args.compare_all_layers:
        _compare_all_layers(
            args.fp16_dir,
            args.quant_dir,
            top_n=args.compare_all_top,
            verbose=args.compare_all_verbose,
            limit=args.compare_all_limit,
            weight_dtype=weight_dtype,
        )
        if args.skip_logits and not (args.twin_compare or args.twin_breakpoint):
            return

    device = _device_from_arg(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer_dir = args.tokenizer_dir or args.fp16_dir

    if args.compare_module_weights:
        skip_modules = {s.strip() for s in args.qat_skip_modules.split(",") if s.strip()}
        bits = int(args.qat_bits)
        group_size = args.qat_group_size
        try:
            cfg = _load_quantize_config(args.quant_dir)
        except FileNotFoundError:
            cfg = None
        if cfg:
            cfg_bits = cfg.get("bits")
            if cfg_bits is not None and "--qat-bits" not in sys.argv:
                bits = int(cfg_bits)
            cfg_group_size = cfg.get("group_size")
            if cfg_group_size is not None and "--qat-group-size" not in sys.argv:
                group_size = int(cfg_group_size)
        qat_weight_dtype = dtype
        if args.weight_dtype is not None:
            qat_weight_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[args.weight_dtype]
        _compare_module_dequant_weights(
            fp16_dir=args.fp16_dir,
            quant_dir=args.quant_dir,
            device=device,
            dtype=qat_weight_dtype,
            dtype_name=args.dtype,
            bits=bits,
            group_size=group_size,
            quant_type=args.qat_quant_type,
            skip_modules=skip_modules,
            limit=args.compare_module_limit,
            verbose=args.compare_module_verbose,
        )
        if args.skip_logits and not (args.twin_compare or args.twin_breakpoint):
            return

    # 2) QAT toggles: compare fake-quant logits vs fp16 logits in the same model.
    qat_model = None
    tokenizer = None
    input_ids = None
    attention_mask = None
    if args.qat_compare or args.compare_qattq:
        group_size = args.qat_group_size or _infer_group_size_from_ckpt(args.fp16_dir)
        skip_modules = {s.strip() for s in args.qat_skip_modules.split(",") if s.strip()}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=True)
        qat_model = _load_qat_model(
            args.fp16_dir,
            device=device,
            dtype=dtype,
            bits=int(args.qat_bits),
            group_size=int(group_size),
            quant_type=args.qat_quant_type,
            skip_modules=skip_modules,
        )
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        from EfficientQAT.core.linear.int_quant_linear import set_quant_state

        with torch.no_grad():
            set_quant_state(qat_model, weight_quant=False)
            logits_fp = qat_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].detach().cpu()
            set_quant_state(qat_model, weight_quant=True)
            logits_q = qat_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].detach().cpu()

        _logits_stats("qat_fp16", logits_fp)
        _logits_stats("qat_fake_quant", logits_q)
        _compare_logits(logits_fp, logits_q, topk=args.topk)

        if args.qat_only and not args.compare_qattq:
            return

    # 3) Compare Triton quant logits vs QAT fake-quant logits.
    if args.compare_qattq:
        if qat_model is None or tokenizer is None:
            raise RuntimeError("QAT model not initialized for --compare-qattq")
        quant_model, _ = _load_quant(args.quant_dir, device, args.dtype)
        with torch.no_grad():
            from EfficientQAT.core.linear.int_quant_linear import set_quant_state

            set_quant_state(qat_model, weight_quant=True)
            qat_logits = qat_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].detach().cpu()
            quant_logits = quant_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].detach().cpu()

        _logits_stats("qat_fake_quant", qat_logits)
        _logits_stats("triton_quant", quant_logits)
        _compare_logits(qat_logits, quant_logits, topk=args.topk)
        if args.qat_only:
            return

    # 4) Default: compare fp16 logits vs Triton quant logits.
    fp16_model, tokenizer = _load_fp16(args.fp16_dir, tokenizer_dir, device, dtype)
    quant_model, quant_tokenizer = _load_quant(args.quant_dir, device, args.dtype)

    if getattr(tokenizer, "vocab_size", None) != getattr(quant_tokenizer, "vocab_size", None):
        print(
            f"warning: tokenizer vocab mismatch fp16={getattr(tokenizer, 'vocab_size', None)} "
            f"quant={getattr(quant_tokenizer, 'vocab_size', None)}"
        )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        fp16_out = fp16_model(input_ids=input_ids, attention_mask=attention_mask)
        quant_out = quant_model(input_ids=input_ids, attention_mask=attention_mask)

    fp16_logits = fp16_out.logits[:, -1, :].detach().cpu()
    quant_logits = quant_out.logits[:, -1, :].detach().cpu()

    _logits_stats("fp16", fp16_logits)
    _logits_stats("quant", quant_logits)
    _compare_logits(fp16_logits, quant_logits, topk=args.topk)

    if args.twin_compare or args.twin_breakpoint:
        include_types = [torch.nn.Linear]
        try:
            from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear

            include_types.append(IntQuantLinear)
        except Exception:
            pass
        try:
            from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear

            include_types.append(TritonV2QuantLinear)
        except Exception:
            pass

        twin_kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            twin_kwargs["attention_mask"] = attention_mask

        cfg = TwinCompareConfig(
            include_types=tuple(include_types),
            include_name_regex=args.twin_include_regex,
            exclude_name_regex=args.twin_exclude_regex,
            save_outputs=True,
            save_outputs_on_types=tuple(include_types),
            mode=args.twin_mode,
            logits_compare=True,
            per_position=True,
            topk=args.topk,
            dump_dir=args.twin_dump_dir,
            breakpoint_threshold=args.twin_breakpoint_threshold,
        )
        report = compare_models(
            fp16_model,
            quant_model,
            example_inputs=(),
            example_kwargs=twin_kwargs,
            cfg=cfg,
        )
        print_compare_report(report, topk=args.twin_topk, metric=args.twin_metric)

        if args.twin_breakpoint:
            bp_report = find_irreversible_breakpoint(
                fp16_model,
                quant_model,
                example_inputs=(),
                example_kwargs=twin_kwargs,
                cfg=cfg,
                strategy=args.twin_breakpoint_strategy,
            )
            print(
                f"breakpoint: baseline_rel_l2={bp_report.baseline_distance:.4g} "
                f"found_index={bp_report.found_index} found_key={bp_report.found_key}"
            )


if __name__ == "__main__":
    main()
