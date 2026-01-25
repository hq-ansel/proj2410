#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Sequence


def _parse_dtype(s: str):
    import torch

    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _parse_pack_dtype(s: str):
    import torch

    if s == "int32":
        return torch.int32
    if s == "int16":
        return torch.int16
    if s == "int8":
        return torch.int8
    raise ValueError(f"Unsupported pack_dtype: {s}")


def _set_module(root, name: str, new_module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _iter_state_dict_files(model_dir: str) -> List[str]:
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
    from safetensors.torch import load_file

    state: Dict[str, "torch.Tensor"] = {}
    for path in _iter_state_dict_files(model_dir):
        state.update(load_file(path, device="cpu"))
    return state


def _discover_quantized_prefixes(state_dict: Dict[str, "torch.Tensor"]) -> List[str]:
    prefixes = []
    for k in state_dict.keys():
        if k.endswith(".qweight"):
            prefixes.append(k[: -len(".qweight")])
    prefixes.sort()
    return prefixes


def _matches_any(name: str, patterns: List[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def _parse_cuda_ids(value: Optional[str]) -> List[int]:
    if value is None or value == "":
        return []
    ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return sorted(set(ids))


def _infer_device_map(model, cuda_ids: Sequence[int], max_memory_ratio: float):
    import torch
    import accelerate

    if not cuda_ids:
        return None
    max_memory = {}
    for idx in cuda_ids:
        try:
            total = torch.cuda.mem_get_info(idx)[1]
        except Exception:
            total = None
        if total is None:
            # fallback: assume 24GiB if CUDA query fails
            max_memory[idx] = "24GiB"
        else:
            max_gib = int(total * max_memory_ratio / (1 << 30))
            max_memory[idx] = f"{max(1, max_gib)}GiB"

    no_split = getattr(model, "_no_split_modules", None) or []
    device_map = accelerate.infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split,
    )
    return device_map


def load_tritonv2_quantized_model(
    model_dir: str,
    device: str = "cuda",
    dtype: str = "float16",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    exclude_patterns: Optional[List[str]] = None,
    cuda_ids: Optional[List[int]] = None,
    max_memory_ratio: float = 0.9,
    use_device_map: bool = True,
):
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from EfficientQAT.core.linear.q_linear_tritonv2 import TritonV2QuantLinear

    model_dir = os.path.abspath(model_dir)
    exclude_patterns = exclude_patterns or []

    cfg_path = os.path.join(model_dir, "quantize_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            qcfg = json.load(f)
        bits = int(qcfg["bits"])
        group_size = int(qcfg["group_size"]) if qcfg.get("group_size") is not None else None
        pack_dtype = _parse_pack_dtype(qcfg.get("pack_dtype", "int32"))
        converted = qcfg.get("converted_modules") or []
        if qcfg.get("excluded_patterns"):
            exclude_patterns = list(exclude_patterns) + list(qcfg["excluded_patterns"])
    else:
        qcfg = None
        bits = None
        group_size = None
        pack_dtype = torch.int32
        converted = []

    state = _load_state_dict(model_dir)
    prefixes = converted or _discover_quantized_prefixes(state)
    if exclude_patterns:
        prefixes = [p for p in prefixes if not _matches_any(p, exclude_patterns)]

    if not prefixes:
        raise RuntimeError("No quantized modules found (missing `*.qweight` keys).")

    torch_dtype = _parse_dtype(dtype)
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    try:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        model = model.to(dtype=torch_dtype)
    model.eval()

    # Replace modules before loading weights
    for prefix in prefixes:
        # Retrieve current module to get shapes
        try:
            mod = model.get_submodule(prefix)
        except AttributeError:
            # PyTorch older than 1.9
            mod = model
            for part in prefix.split("."):
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"{prefix}: expected nn.Linear in fresh model, got {type(mod)}")

        if bits is None:
            raise RuntimeError("Missing quantize_config.json (bits/group_size/pack_dtype are required).")

        qlinear = TritonV2QuantLinear(
            bits=bits,
            group_size=group_size if group_size is not None else mod.in_features,
            desc_act=False,
            sym=False,
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=mod.bias is not None,
            pack_dtype=pack_dtype,
        )
        qlinear.post_init()
        _set_module(model, prefix, qlinear)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading quantized checkpoint: {unexpected[:20]}")
    if missing:
        raise RuntimeError(f"Missing keys when loading quantized checkpoint: {missing[:20]}")

    if use_device_map and device.startswith("cuda") and torch.cuda.device_count() > 1:
        if cuda_ids is None:
            cuda_ids = list(range(torch.cuda.device_count()))
        device_map = _infer_device_map(model, cuda_ids=cuda_ids, max_memory_ratio=max_memory_ratio)
        if device_map:
            import accelerate

            model = accelerate.dispatch_model(model, device_map=device_map)
        else:
            model.to(device)
    else:
        model.to(device)
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Quantized checkpoint dir (contains quantize_config.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--cuda-ids",
        default=None,
        help="Comma-separated CUDA device ids to use for device_map (default: all visible).",
    )
    ap.add_argument(
        "--max-memory-ratio",
        type=float,
        default=0.9,
        help="Per-GPU memory ratio for device_map inference.",
    )
    ap.add_argument("--no-device-map", action="store_true", help="Disable device_map sharding.")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    args = ap.parse_args()

    cuda_ids = None
    if args.cuda_ids is not None:
        cuda_ids = _parse_cuda_ids(args.cuda_ids)

    model, tokenizer = load_tritonv2_quantized_model(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        cuda_ids=cuda_ids,
        max_memory_ratio=args.max_memory_ratio,
        use_device_map=not args.no_device_map,
    )
    print("Loaded model:", type(model))
    print("Tokenizer vocab size:", getattr(tokenizer, "vocab_size", None))


"""
python3 VeOmni/tasks/quantize/load_tritonv2_quant.py   \
    --model-dir /home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/test/out  \
    --device cuda   \
    --dtype bfloat16
"""

if __name__ == "__main__":
    main()
