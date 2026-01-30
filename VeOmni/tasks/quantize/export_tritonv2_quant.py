#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _parse_dtype(s: str):
    """
    Parse dtype string to torch dtype.

    Args:
        s: Dtype string ("float16", "bfloat16", "float32", "auto")

    Returns:
        torch.dtype | None: Corresponding torch dtype, or None for "auto"

    Raises:
        ValueError: If dtype is not supported
    """
    import torch

    if s == "auto":
        return None
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _parse_pack_dtype(s: str):
    """
    Parse pack dtype string to torch dtype.

    Args:
        s: Pack dtype string ("int32", "int16", "int8")

    Returns:
        torch.dtype: Corresponding torch dtype

    Raises:
        ValueError: If dtype is not supported
    """
    import torch

    if s == "int32":
        return torch.int32
    if s == "int16":
        return torch.int16
    if s == "int8":
        return torch.int8
    raise ValueError(f"Unsupported pack_dtype: {s}")


def _iter_state_dict_files(model_dir: str) -> List[str]:
    """
    Return a list of safetensors shard file paths in load order.

    Supports:
      - Single file: model.safetensors
      - Sharded: model.safetensors.index.json + multiple shard files

    Args:
        model_dir: Directory containing checkpoint files

    Returns:
        List of absolute paths to safetensors files

    Raises:
        FileNotFoundError: if no checkpoint files found
    """
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


def _load_state_dict(model_dir: str) -> Dict[str, "torch.Tensor"]:
    """
    Load state_dict from safetensors checkpoint files.

    Supports both single file (model.safetensors) and sharded checkpoints
    (model.safetensors.index.json + shards).

    Args:
        model_dir: Directory containing checkpoint files

    Returns:
        Dict mapping tensor names to CPU tensors
    """
    import torch
    from safetensors.torch import load_file

    state: Dict[str, torch.Tensor] = {}
    for path in _iter_state_dict_files(model_dir):
        state.update(load_file(path, device="cpu"))
    return state


def _infer_weight_dtype_from_state(state_dict: Dict[str, "torch.Tensor"]) -> "torch.dtype":
    """
    Infer weight dtype from a loaded state_dict by picking the first floating tensor.
    Falls back to float16 if no floating tensors are present.
    """
    import torch

    for tensor in state_dict.values():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return torch.float16


def _save_state_dict(model_dir: str, state_dict: Dict[str, "torch.Tensor"]) -> None:
    """
    Save state_dict to a single safetensors file.

    Args:
        model_dir: Output directory
        state_dict: Dict mapping tensor names to tensors (can be on any device)
    """
    from safetensors.torch import save_file

    os.makedirs(model_dir, exist_ok=True)
    # safetensors requires contiguous CPU tensors
    to_save = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
    save_file(to_save, os.path.join(model_dir, "model.safetensors"), metadata={"format": "pt"})


def _copy_assets(src_dir: str, dst_dir: str) -> None:
    """
    Copy config/tokenizer assets from source to destination.

    Skips weight files and index files since we write new quantized weights.
    Preserves all other files and directories (config.json, tokenizer files, etc.).

    Args:
        src_dir: Source checkpoint directory
        dst_dir: Destination checkpoint directory
    """
    os.makedirs(dst_dir, exist_ok=True)
    skip = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "quantize_config.json",
    }
    for name in os.listdir(src_dir):
        if name in skip:
            continue
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def _should_exclude(module_name: str, exclude_patterns: List[str]) -> bool:
    """
    Check if a module name matches any exclusion patterns.

    Args:
        module_name: Module prefix (e.g., "model.layers.0.mlp.gate_proj")
        exclude_patterns: List of regex patterns

    Returns:
        True if module matches any pattern, False otherwise
    """
    return any(re.search(p, module_name) for p in exclude_patterns)


@dataclass(frozen=True)
class LinearQatParams:
    weight: "torch.Tensor"  # [out_features, in_features]
    bias: Optional["torch.Tensor"]  # [out_features] or None
    scale: "torch.Tensor"  # [num_groups] or [num_groups, 1] - flattened per-group quantization scales
    zero_point: Optional["torch.Tensor"]  # [num_groups] or [num_groups, 1] - flattened per-group zero points
    symmetric: bool = False


def _find_qat_linear_prefixes(state_dict: Dict[str, "torch.Tensor"]) -> List[str]:
    """
    Find all quantized linear module prefixes in the state_dict.

    IntQuantLinear modules have "*.weight_quantizer.scale" keys in the state_dict.
    This function extracts the module prefixes from these keys.

    Args:
        state_dict: model state_dict

    Returns:
        List of module prefixes (e.g., ["model.layers.0.mlp.gate_proj", ...])
    """
    prefixes = []
    for k in state_dict.keys():
        if k.endswith(".weight_quantizer.scale"):
            prefixes.append(k[: -len(".weight_quantizer.scale")])
    prefixes.sort()
    return prefixes


def _extract_qat_params(state_dict: Dict[str, "torch.Tensor"], prefix: str) -> LinearQatParams:
    """
    Extract quantization parameters for a single linear layer from state_dict.
    
    Args:
        state_dict: model state_dict containing quantized weights
        prefix: module prefix (e.g., "model.layers.0.mlp.gate_proj")
    
    Returns:
        LinearQatParams with:
            weight: [out_features, in_features]
            bias: [out_features] or None
            scale: [num_groups] or [num_groups, 1]
            zero_point: [num_groups] or [num_groups, 1]
    """
    import torch

    w_key = f"{prefix}.weight"
    if w_key not in state_dict:
        raise KeyError(f"Missing {w_key} for quantized module {prefix}")
    weight = state_dict[w_key]
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight for {prefix}, got shape={tuple(weight.shape)}")

    bias = state_dict.get(f"{prefix}.bias")
    # [out_features*in_features/group_size, group_size]
    scale = state_dict.get(f"{prefix}.weight_quantizer.scale")
    zero_point = state_dict.get(f"{prefix}.weight_quantizer.zero_point")
    if scale is None:
        raise KeyError(f"Missing quantizer params for {prefix}: scale")

    scale = scale.detach().to(device="cpu")
    if scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        scale = scale.float()
    symmetric = zero_point is None
    if zero_point is not None:
        zero_point = zero_point.detach().to(device="cpu")
        if zero_point.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64):
            zero_point = zero_point.float()

    return LinearQatParams(
        weight=weight.detach().to("cpu"),
        bias=bias,
        scale=scale,
        zero_point=zero_point,
        symmetric=symmetric,
    )


def _infer_group_size(in_features: int, out_features: int, scale_numel: int) -> int:
    """
    Infer group_size from QAT quantizer parameters.

    IntQuantLinear's quantizer initializes scale/zero_point by `weight.reshape(-1, group_size)`,
    so total groups should be (out_features * in_features) / group_size when divisible.
    
    Args:
        in_features: input dimension [in_features]
        out_features: output dimension [out_features]
        scale_numel: number of scale values = total_groups = out_features * in_features / group_size
    
    Returns:
        group_size: size of each quantization group along in_features dimension
    """
    total = in_features * out_features
    if scale_numel <= 0 or total % scale_numel != 0:
        raise ValueError(
            f"Cannot infer group_size: in={in_features}, out={out_features}, scale_numel={scale_numel}"
        )
    return total // scale_numel


# def _reshape_qat_scale_zp(
#     scale: "torch.Tensor",  # [num_groups] - flattened scales
#     zero_point: "torch.Tensor",  # [num_groups] - flattened zero points
#     out_features: int,
#     in_features: int,
#     group_size: int
# ) -> Tuple["torch.Tensor", "torch.Tensor"]:
#     """
#     Convert QAT quantizer params (flattened groups) into q_linear pack format: [out_features, n_groups_in].
    
#     Returns:
#         scales_out_g: [out_features, n_groups_in] - reshaped scales per output channel
#         zeros_out_g_f: [out_features, n_groups_in] - reshaped zero points per output channel
#     """
#     import torch

#     if group_size == -1 or group_size > in_features:
#         group_size = in_features
#     if in_features % group_size != 0:
#         raise ValueError(
#             f"Expected in_features divisible by group_size for QAT param reshape: in={in_features}, group={group_size}"
#         )
#     n_groups_in = in_features // group_size
#     expected = out_features * n_groups_in

#     s = scale.reshape(-1)
#     z = zero_point.reshape(-1)
#     if s.numel() != expected or z.numel() != expected:
#         raise ValueError(
#             f"Unexpected quant param numel for reshape: scale={s.numel()}, zp={z.numel()}, expected={expected} "
#             f"(out={out_features}, in={in_features}, group={group_size})"
#         )

#     s = s.view(out_features, n_groups_in).to(dtype=torch.float16)
#     z = z.view(out_features, n_groups_in).to(dtype=torch.float32)
#     return s, z

def _reshape_qat_scale_zp(scale, zero_point, out_features, in_features, group_size, order="auto"):
    import torch
    if group_size == -1 or group_size > in_features:
        group_size = in_features
    n_groups_in = in_features // group_size
    expected = out_features * n_groups_in

    def to_out_g(t, name):
        # squeeze 常见 [...,1]
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        # 2D：直接判定
        if t.ndim == 2:
            if t.shape == (out_features, n_groups_in):
                return t
            if t.shape == (n_groups_in, out_features):
                return t.T
            raise ValueError(f"{name} 2D shape unexpected: {tuple(t.shape)}")

        # 1D：两种解释都可能
        if t.numel() != expected:
            raise ValueError(f"{name} numel={t.numel()} expected={expected}")

        flat = t.reshape(-1)
        if order == "out_major":
            return flat.view(out_features, n_groups_in)
        if order == "group_major":
            return flat.view(n_groups_in, out_features).T
        if order == "auto":
            # 先给出一个默认（你也可以在外面用对比来自动选择）
            return flat.view(out_features, n_groups_in)

        raise ValueError(f"unknown order={order}")

    s = to_out_g(scale, "scale").to(torch.float16)
    if zero_point is None:
        z = torch.zeros_like(s, dtype=torch.float32)
    else:
        z = to_out_g(zero_point, "zero_point").to(torch.float32)
    return s, z


def _rescale_symmetric_scale(scale: "torch.Tensor", bits: int) -> "torch.Tensor":
    if bits <= 1:
        return scale
    # QAT symmetric uses qmax=2^bits-1; Triton packed signed uses qmax=2^(bits-1)-1.
    num = (1 << bits) - 1
    denom = (1 << (bits - 1)) - 1
    if denom <= 0:
        return scale
    return scale * (float(num) / float(denom))



def _clamp_qparams(
    scale: "torch.Tensor",  # [out_features, n_groups_in] - reshaped scales
    zero_point: "torch.Tensor",  # [out_features, n_groups_in] - reshaped zero points (float)
    bits: int
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Clamp quantization parameters to valid ranges to prevent packing errors.
    
    Args:
        scale: scales to clamp, shape [out_features, n_groups_in]
        zero_point: zero points to clamp, shape [out_features, n_groups_in]
        bits: quantization bits (2/4/8)
    
    Returns:
        clamped_scale: [out_features, n_groups_in] - clamped to [1e-5, 1e4] range
        clamped_zero_point: [out_features, n_groups_in] - clamped to [0, 2^bits-1] range
    """
    import torch

    min_scale = 1e-5
    max_scale = 1e4
    sign = torch.where(scale >= 0, torch.ones_like(scale), -torch.ones_like(scale))
    scale = torch.clamp(scale.abs(), min_scale, max_scale).to(dtype=scale.dtype) * sign

    maxq = (1 << bits) - 1
    zero_point = torch.clamp(torch.round(zero_point), 0, maxq).to(dtype=zero_point.dtype)
    return scale, zero_point


def _pack_one_linear(
    prefix: str,
    qat: LinearQatParams,  # weight: [out_features, in_features], scale/zero_point: [num_groups]
    bits: int,
    group_size: int,
    pack_dtype: "torch.dtype",
    weight_dtype: "torch.dtype",
    qat_param_order: str = "auto",
    return_dequant: bool = False,
) -> Tuple[Dict[str, "torch.Tensor"], Optional["torch.nn.Module"]]:
    """
    Pack a single linear layer's weights into tritonv2 quantized format.
    
    Returns:
        (packed, qlinear):
        packed keys:
            - "{prefix}.qweight": [out_features // bits * pack_dtype_size, in_features] - packed weights
            - "{prefix}.qzeros": [out_features // bits * pack_dtype_size, in_features // group_size] - packed zeros
            - "{prefix}.scales": [out_features, in_features // group_size] - scales per output channel and group
            - "{prefix}.g_idx": [in_features] - group index mapping
            - "{prefix}.bias": [out_features] - bias (optional)
        qlinear: TritonV2QuantLinear instance if return_dequant=True, else None
    """
    import torch
    import torch.nn as nn

    from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear

    out_features, in_features = qat.weight.shape
    if in_features % 32 != 0 or out_features % 32 != 0:
        raise ValueError(
            f"{prefix}: TritonV2QuantLinear requires in/out divisible by 32, got in={in_features}, out={out_features}"
        )

    inferred_group = _infer_group_size(in_features, out_features, qat.scale.numel())
    if group_size is None:
        group_size = inferred_group
    if inferred_group != group_size:
        raise ValueError(
            f"{prefix}: inferred group_size={inferred_group} from QAT params but CLI group_size={group_size}"
        )

    # Reshape: [num_groups] -> [out_features, n_groups_in] where n_groups_in = in_features // group_size
    scales_out_g, zeros_out_g_f = _reshape_qat_scale_zp(
        qat.scale,
        qat.zero_point,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        order=qat_param_order,
    )
    if qat.symmetric:
        scales_out_g = _rescale_symmetric_scale(scales_out_g, bits=bits)
        zeros_out_g_f = torch.zeros_like(scales_out_g, dtype=torch.float32)
    # Match training-time clamping (cal_qparams) to avoid out-of-range pack values
    scales_out_g, zeros_out_g_f = _clamp_qparams(scales_out_g, zeros_out_g_f, bits=bits)
    if scales_out_g.ndim != 2 or zeros_out_g_f.ndim != 2:
        raise RuntimeError(
            f"{prefix}: expected 2D scales/zeros after reshape, got scales={tuple(scales_out_g.shape)}, zeros={tuple(zeros_out_g_f.shape)}"
        )

    maxq = (1 << bits) - 1
    # Convert zero points from float to int32: [out_features, n_groups_in] -> [out_features, n_groups_in] (int32)
    zeros_out_g = torch.round(zeros_out_g_f).clamp(0, maxq).to(dtype=torch.int32)

    # Hard assertions for safety before packing
    assert torch.isfinite(scales_out_g).all(), f"{prefix}: scales contain NaN or Inf values"
    assert zeros_out_g.dtype == torch.int32, f"{prefix}: zeros dtype must be int32, got {zeros_out_g.dtype}"
    assert int(zeros_out_g.min()) >= 0, f"{prefix}: zeros contain negative values (min={int(zeros_out_g.min())})"
    assert int(zeros_out_g.max()) <= maxq, f"{prefix}: zeros exceed max value (max={int(zeros_out_g.max())}, maxq={maxq})"

    # Create nn.Linear with float weights: weight [out_features, in_features], bias [out_features]
    linear = nn.Linear(in_features, out_features, bias=qat.bias is not None)
    linear.weight.data.copy_(qat.weight.to(dtype=weight_dtype))
    if qat.bias is not None and linear.bias is not None:
        linear.bias.data.copy_(qat.bias.to(dtype=weight_dtype))

    # Create TritonV2QuantLinear: will hold packed quantized weights
    qlinear = TritonV2QuantLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=qat.symmetric,
        in_features=in_features,
        out_features=out_features,
        bias=qat.bias is not None,
        pack_dtype=pack_dtype,
    )
    qlinear.post_init()
    # Pack: converts float weights to packed quantized format
    qlinear.pack(linear=linear, scales=scales_out_g, zeros=zeros_out_g)

    # Packed output tensors for tritonv2 kernel
    packed: Dict[str, torch.Tensor] = {
        f"{prefix}.qweight": qlinear.qweight,  # packed quantized weights
        f"{prefix}.qzeros": qlinear.qzeros,   # packed zero points
        f"{prefix}.scales": qlinear.scales,   # per-group scales
        f"{prefix}.g_idx": qlinear.g_idx,     # group index mapping
    }
    if qlinear.bias is not None:
        packed[f"{prefix}.bias"] = qlinear.bias
    if return_dequant:
        return packed, qlinear
    return packed, None


def _drop_quantizer_keys(state_dict: Dict[str, "torch.Tensor"], prefix: str) -> None:
    """
    Remove quantizer parameters from state_dict for a given module.

    Drops keys like "{prefix}.weight_quantizer.scale" and "{prefix}.weight_quantizer.zero_point".

    Args:
        state_dict: Model state_dict to modify in-place
        prefix: Module prefix (e.g., "model.layers.0.mlp.gate_proj")
    """
    to_drop = [k for k in state_dict.keys() if k.startswith(f"{prefix}.weight_quantizer.")]
    for k in to_drop:
        state_dict.pop(k, None)


def _drop_triton_keys(state_dict: Dict[str, "torch.Tensor"], prefix: str) -> None:
    for suffix in (".qweight", ".qzeros", ".scales", ".g_idx"):
        state_dict.pop(f"{prefix}{suffix}", None)


def main() -> None:
    ap = argparse.ArgumentParser()
    default_root = "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g64-int2-kd/checkpoints"
    ap.add_argument(
        "--src",
        default=os.path.join(default_root, "global_step_614", "hf_ckpt"),
        help="Input HF checkpoint dir (contains model.safetensors*)",
    )
    ap.add_argument(
        "--dst",
        default=os.path.join(default_root, "out"),
        help="Output dir for tritonv2 quantized checkpoint",
    )
    ap.add_argument(
        "--dst-dequant",
        default=os.path.join(default_root, "out_dequant"),
        help="Optional output dir for dequantized checkpoint (defaults to <dst>_dequant unless --skip-dequant).",
    )
    ap.add_argument("--skip-dequant", action="store_true", help="Skip writing dequantized checkpoint.")
    ap.add_argument(
        "--veomni-yaml",
        default=None,
        help="Optional VeOmni YAML (with `quantizer.n_bits/group_size`) to fill missing CLI args.",
    )
    ap.add_argument("--bits", type=int, default=2, help="Quant bits for TritonV2QuantLinear (2/4/8)")
    ap.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size along in_features; default: infer from QAT scale shape (requires in%group==0).",
    )
    ap.add_argument("--pack-dtype", default="int32", choices=["int32", "int16", "int8"])
    ap.add_argument(
        "--weight-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Weight dtype used for packing; auto infers from src weights.",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Regex pattern; if matches module prefix, skip conversion (can be repeated).",
    )
    ap.add_argument(
        "--qat-param-order",
        default="out_major",
        choices=["out_major", "group_major"],
        help="Order to reshape QAT scale/zero_point into [out_features, n_groups_in].",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="After export, compare src vs dequant vs packed-dequant (first N layers).",
    )
    ap.add_argument(
        "--compare-max-layers",
        type=int,
        default=20,
        help="Max layers to compare when --compare is set.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Scan & report without writing output")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)
    dst_dequant = None
    if not args.skip_dequant:
        dst_dequant = os.path.abspath(args.dst_dequant) if args.dst_dequant else f"{dst_dir}_dequant"

    if args.veomni_yaml is not None:
        import yaml

        with open(os.path.abspath(args.veomni_yaml), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        q = cfg.get("quantizer", {}) if isinstance(cfg, dict) else {}
        if args.bits is None and "n_bits" in q:
            args.bits = int(q["n_bits"])
        if args.group_size is None and "group_size" in q:
            args.group_size = int(q["group_size"])

    if args.bits is None:
        raise ValueError("`--bits` is required (or provide `--veomni-yaml` with quantizer.n_bits).")

    import torch

    pack_dtype = _parse_pack_dtype(args.pack_dtype)

    # Load state_dict: {tensor_name: tensor} where tensors are on CPU
    state = _load_state_dict(src_dir)
    weight_dtype = _infer_weight_dtype_from_state(state) if args.weight_dtype == "auto" else _parse_dtype(args.weight_dtype)
    # Find all quantized linear module prefixes (e.g., "model.layers.0.mlp.gate_proj")
    prefixes = _find_qat_linear_prefixes(state)
    if args.exclude:
        prefixes = [p for p in prefixes if not _should_exclude(p, args.exclude)]

    if not prefixes:
        raise RuntimeError(
            "No quantized IntQuantLinear modules found in checkpoint (missing `*.weight_quantizer.scale` keys)."
        )

    if args.dry_run:
        print(f"Found {len(prefixes)} quantized linear modules.")
        print("Examples:")
        for p in prefixes[:20]:
            print(" -", p)
        return

    # Output state_dict with packed quantized weights
    new_state: Dict[str, torch.Tensor] = dict(state)
    dequant_state: Optional[Dict[str, torch.Tensor]] = dict(state) if dst_dequant else None
    # List of successfully converted module prefixes
    converted: List[str] = []
    # List of (prefix, error_reason) for skipped modules
    skipped: List[Tuple[str, str]] = []
    # Resolved group_size (inferred from first module if not provided)
    resolved_group_size: Optional[int] = args.group_size
    sym_by_module: Dict[str, bool] = {}
    sym_all = True

    for prefix in prefixes:
        qat = _extract_qat_params(state, prefix)
        try:
            packed, qlinear = _pack_one_linear(
                prefix=prefix,
                qat=qat,
                bits=int(args.bits),
                group_size=args.group_size,
                pack_dtype=pack_dtype,
                weight_dtype=weight_dtype,
                qat_param_order=args.qat_param_order,
                return_dequant=dequant_state is not None,
            )
        except Exception as e:
            # Fallback: keep float weight/bias, but drop quantizer params so the layer can be loaded as nn.Linear.
            _drop_quantizer_keys(new_state, prefix)
            skipped.append((prefix, f"{type(e).__name__}: {e}"))
            continue

        # Drop original float weights (now replaced by packed buffers) + quantizer params.
        new_state.pop(f"{prefix}.weight", None)
        new_state.pop(f"{prefix}.bias", None)
        _drop_quantizer_keys(new_state, prefix)
        new_state.update(packed)
        converted.append(prefix)
        sym_by_module[prefix] = bool(qat.symmetric)
        sym_all = sym_all and bool(qat.symmetric)

        if dequant_state is not None and qlinear is not None:
            dequant_weight = qlinear.dequantize_weight().to(dtype=weight_dtype)
            dequant_state[f"{prefix}.weight"] = dequant_weight.detach().cpu()
            if qlinear.bias is not None:
                dequant_state[f"{prefix}.bias"] = qlinear.bias.detach().cpu()
            _drop_quantizer_keys(dequant_state, prefix)
            _drop_triton_keys(dequant_state, prefix)

        # Record resolved group_size (inferred if not provided)
        if resolved_group_size is None:
            # Infer from packed g_idx: g_idx increases every group_size positions.
            g_idx = packed[f"{prefix}.g_idx"].to(dtype=torch.int64)
            # Find first index where g_idx becomes 1; if none, group_size == in_features.
            idx = (g_idx == 1).nonzero(as_tuple=True)[0]
            resolved_group_size = int(idx[0].item()) if idx.numel() else int(g_idx.numel())

    os.makedirs(dst_dir, exist_ok=True)
    _copy_assets(src_dir, dst_dir)
    _save_state_dict(dst_dir, new_state)
    if dequant_state is not None:
        os.makedirs(dst_dequant, exist_ok=True)
        _copy_assets(src_dir, dst_dequant)
        _save_state_dict(dst_dequant, dequant_state)

    cfg = {
        "quant_type": "tritonv2",
        "bits": int(args.bits),
        "group_size": int(resolved_group_size) if resolved_group_size is not None else None,
        "pack_dtype": args.pack_dtype,
        "converted_modules": converted,
        "skipped_modules": [{"name": n, "reason": r} for n, r in skipped],
        "excluded_patterns": args.exclude,
        "sym": bool(sym_all) if converted else False,
        "sym_by_module": sym_by_module,
    }
    with open(os.path.join(dst_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"Saved tritonv2 quantized checkpoint to: {dst_dir}")
    print(f"Converted modules: {len(converted)}")
    if dequant_state is not None:
        print(f"Saved dequantized checkpoint to: {dst_dequant}")

    if args.compare:
        try:
            from safetensors.torch import load_file
            from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear
        except Exception as e:
            raise RuntimeError(f"Compare requires safetensors + TritonV2QuantLinear: {e}")

        def load_state(model_dir: str) -> Dict[str, "torch.Tensor"]:
            single = os.path.join(model_dir, "model.safetensors")
            if os.path.isfile(single):
                return load_file(single, device="cpu")
            index_path = os.path.join(model_dir, "model.safetensors.index.json")
            if os.path.isfile(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                shards = sorted(set(weight_map.values()))
                state = {}
                for name in shards:
                    state.update(load_file(os.path.join(model_dir, name), device="cpu"))
                return state
            raise FileNotFoundError(f"Missing safetensors under {model_dir}")

        def mse(a, b) -> float:
            return float((a - b).float().pow(2).mean().item())

        def mae(a, b) -> float:
            return float((a - b).float().abs().mean().item())

        def max_err(a, b) -> float:
            return float((a - b).float().abs().max().item())

        def _shape_str(t: "torch.Tensor") -> str:
            return "x".join(str(s) for s in t.shape)

        def _align_for_compare(a: "torch.Tensor", b: "torch.Tensor", label: str):
            if a.shape == b.shape:
                return b, ""
            if a.ndim == 2 and b.ndim == 2 and a.shape == (b.shape[1], b.shape[0]):
                return b.t(), " (transposed)"
            return None, f"shape mismatch src={_shape_str(a)} {label}={_shape_str(b)}"

        def _fmt_metrics(label: str, note: str, a: "torch.Tensor", b: "torch.Tensor") -> str:
            return (
                f"  {label}{note}: "
                f"mse={mse(a,b):.3e} mae={mae(a,b):.3e} max={max_err(a,b):.3e}"
            )

        src_state = load_state(src_dir)
        deq_state = load_state(dst_dequant)
        out_state = load_state(dst_dir)

        cfg_path = os.path.join(dst_dir, "quantize_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            qcfg = json.load(f)
        bits = int(qcfg["bits"])
        group_size = int(qcfg["group_size"])
        pack_dtype_name = qcfg.get("pack_dtype", "int32")
        pack_dtype = getattr(torch, pack_dtype_name)
        sym_map = qcfg.get("sym_by_module", {}) or {}
        sym_default = bool(qcfg.get("sym", False))

        def dequant_from_packed(prefix: str) -> "torch.Tensor":
            qlinear = TritonV2QuantLinear(
                bits=bits,
                group_size=group_size,
                desc_act=False,
                sym=bool(sym_map.get(prefix, sym_default)),
                in_features=out_state[f"{prefix}.g_idx"].numel(),
                out_features=out_state[f"{prefix}.scales"].shape[1],
                bias=f"{prefix}.bias" in out_state,
                pack_dtype=pack_dtype,
            )
            qlinear.post_init()
            qlinear.qweight = out_state[f"{prefix}.qweight"]
            qlinear.qzeros = out_state[f"{prefix}.qzeros"]
            qlinear.scales = out_state[f"{prefix}.scales"]
            qlinear.g_idx = out_state[f"{prefix}.g_idx"]
            if qlinear.bias is not None and f"{prefix}.bias" in out_state:
                qlinear.bias = out_state[f"{prefix}.bias"]
            return qlinear.dequantize_weight()

        weight_keys = [k for k in src_state.keys() if k.endswith(".weight") and k in deq_state]
        weight_keys.sort()
        if not weight_keys:
            raise RuntimeError("No shared .weight keys between src and dequant checkpoints.")

        sample = weight_keys[: max(int(args.compare_max_layers), 0)]
        print(f"Comparing {len(sample)} layers (first {len(sample)} by name)...")
        skip_src_deq = 0
        skip_src_packed = 0
        skip_deq_packed = 0
        for k in sample:
            a = src_state[k]
            b = deq_state[k]
            prefix = k[: -len(".weight")]
            b_aligned, b_note = _align_for_compare(a, b, "dequant")
            if b_aligned is None:
                skip_src_deq += 1
                print(f"{k}:\n  src vs dequant: {b_note} (skipped)")
                continue
            if f"{prefix}.qweight" in out_state:
                c = dequant_from_packed(prefix)
                c_aligned, c_note = _align_for_compare(a, c, "packed-dequant")
                if c_aligned is None:
                    skip_src_packed += 1
                    print(
                        "\n".join(
                            [
                                f"{k}:",
                                _fmt_metrics("src vs dequant", b_note, a, b_aligned),
                                f"  src vs packed-dequant: {c_note} (skipped)",
                            ]
                        )
                    )
                    continue
                d_note = ""
                if b_aligned.shape != c_aligned.shape:
                    d_note = (
                        f"shape mismatch dequant={_shape_str(b_aligned)} "
                        f"packed-dequant={_shape_str(c_aligned)}"
                    )
                if d_note:
                    skip_deq_packed += 1
                    deq_line = f"  dequant vs packed-dequant: {d_note} (skipped)"
                else:
                    deq_line = _fmt_metrics("dequant vs packed-dequant", "", b_aligned, c_aligned)
                print(
                    "\n".join(
                        [
                            f"{k}:",
                            _fmt_metrics("src vs dequant", b_note, a, b_aligned),
                            _fmt_metrics("src vs packed-dequant", c_note, a, c_aligned),
                            deq_line,
                        ]
                    )
                )
            else:
                print(
                    "\n".join(
                        [
                            f"{k}:",
                            _fmt_metrics("src vs dequant", b_note, a, b_aligned),
                        ]
                    )
                )
        if skip_src_deq or skip_src_packed or skip_deq_packed:
            print(
                "Compare summary (skipped due to shape mismatch): "
                f"src vs dequant={skip_src_deq}, "
                f"src vs packed-dequant={skip_src_packed}, "
                f"dequant vs packed-dequant={skip_deq_packed}"
            )


def export_tritonv2_quantized_checkpoint(
    *,
    src: str,
    dst: str,
    dst_dequant: Optional[str] = None,
    save_dequant: bool = True,
    bits: int,
    group_size: Optional[int] = None,
    pack_dtype: str = "int32",
    weight_dtype: str = "auto",
    qat_param_order: str = "auto",
    exclude: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Programmatic API for training script integration.

    Converts a QAT checkpoint with IntQuantLinear modules to tritonv2 quantized format.

    Args:
        src: Source checkpoint directory (contains model.safetensors or model.safetensors.index.json)
        dst: Destination directory for packed quantized checkpoint
        bits: Quantization bits (2/4/8)
        group_size: Group size along in_features dimension (optional, inferred from QAT params)
        pack_dtype: Pack data type ("int32", "int16", or "int8")
        weight_dtype: Weight data type for bias storage ("auto", "float16", "bfloat16", or "float32")
        qat_param_order: Order to reshape QAT scale/zero_point ("auto", "out_major", or "group_major")
        exclude: List of regex patterns to exclude module prefixes
        dst_dequant: Optional output dir for dequantized checkpoint
        save_dequant: Whether to write dequantized checkpoint

    Returns:
        Dict with quantization summary (also written to dst/quantize_config.json):
            - quant_type: "tritonv2"
            - bits: number of quantization bits
            - group_size: resolved group size
            - pack_dtype: pack data type
            - converted_modules: list of successfully converted module prefixes
            - skipped_modules: list of {name, reason} for skipped modules
            - excluded_patterns: list of exclusion patterns applied
    """
    import torch

    src_dir = os.path.abspath(src)
    dst_dir = os.path.abspath(dst)
    dst_dequant_dir = None
    if save_dequant:
        dst_dequant_dir = os.path.abspath(dst_dequant) if dst_dequant else f"{dst_dir}_dequant"
    exclude = exclude or []

    pack_dtype_t = _parse_pack_dtype(pack_dtype)

    # Load state_dict: {tensor_name: tensor} where tensors are on CPU
    state = _load_state_dict(src_dir)
    weight_dtype_t = _infer_weight_dtype_from_state(state) if weight_dtype == "auto" else _parse_dtype(weight_dtype)
    # Find all quantized linear module prefixes (e.g., "model.layers.0.mlp.gate_proj")
    prefixes = _find_qat_linear_prefixes(state)
    if exclude:
        prefixes = [p for p in prefixes if not _should_exclude(p, exclude)]
    if not prefixes:
        raise RuntimeError(
            "No quantized IntQuantLinear modules found in checkpoint (missing `*.weight_quantizer.scale` keys)."
        )

    new_state: Dict[str, torch.Tensor] = dict(state)
    dequant_state: Optional[Dict[str, torch.Tensor]] = dict(state) if dst_dequant_dir else None
    converted: List[str] = []
    skipped: List[Tuple[str, str]] = []
    resolved_group_size: Optional[int] = group_size
    sym_by_module: Dict[str, bool] = {}
    sym_all = True

    for prefix in prefixes:
        qat = _extract_qat_params(state, prefix)
        try:
            packed, qlinear = _pack_one_linear(
                prefix=prefix,
                qat=qat,
                bits=int(bits),
                group_size=group_size,
                pack_dtype=pack_dtype_t,
                weight_dtype=weight_dtype_t,
                qat_param_order=qat_param_order,
                return_dequant=dequant_state is not None,
            )
        except Exception as e:
            _drop_quantizer_keys(new_state, prefix)
            skipped.append((prefix, f"{type(e).__name__}: {e}"))
            continue

        new_state.pop(f"{prefix}.weight", None)
        new_state.pop(f"{prefix}.bias", None)
        _drop_quantizer_keys(new_state, prefix)
        new_state.update(packed)
        converted.append(prefix)
        sym_by_module[prefix] = bool(qat.symmetric)
        sym_all = sym_all and bool(qat.symmetric)

        if resolved_group_size is None:
            g_idx = packed[f"{prefix}.g_idx"].to(dtype=torch.int64)
            idx = (g_idx == 1).nonzero(as_tuple=True)[0]
            resolved_group_size = int(idx[0].item()) if idx.numel() else int(g_idx.numel())

        if dequant_state is not None and qlinear is not None:
            dequant_weight = qlinear.dequantize_weight().to(dtype=weight_dtype_t)
            dequant_state[f"{prefix}.weight"] = dequant_weight.detach().cpu()
            if qlinear.bias is not None:
                dequant_state[f"{prefix}.bias"] = qlinear.bias.detach().cpu()
            _drop_quantizer_keys(dequant_state, prefix)
            _drop_triton_keys(dequant_state, prefix)

    os.makedirs(dst_dir, exist_ok=True)
    _copy_assets(src_dir, dst_dir)
    _save_state_dict(dst_dir, new_state)
    if dequant_state is not None:
        os.makedirs(dst_dequant_dir, exist_ok=True)
        _copy_assets(src_dir, dst_dequant_dir)
        _save_state_dict(dst_dequant_dir, dequant_state)

    cfg: Dict[str, object] = {
        "quant_type": "tritonv2",
        "bits": int(bits),
        "group_size": int(resolved_group_size) if resolved_group_size is not None else None,
        "pack_dtype": pack_dtype,
        "converted_modules": converted,
        "skipped_modules": [{"name": n, "reason": r} for n, r in skipped],
        "excluded_patterns": exclude,
        "sym": bool(sym_all) if converted else False,
        "sym_by_module": sym_by_module,
    }
    with open(os.path.join(dst_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return cfg

"""
python3 VeOmni/tasks/quantize/export_tritonv2_quant.py   \
    --src /home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/test/checkpoints/global_step_2450/hf_ckpt  \
    --dst /home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/test/out   \
    --veomni-yaml VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B.yaml   \
    --pack-dtype int32
"""


if __name__ == "__main__":
    main()
