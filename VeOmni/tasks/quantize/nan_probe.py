#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml

from veomni.checkpoint import ckpt_to_state_dict
from veomni.distributed.parallel_state import init_parallel_state
from veomni.models import build_foundation_model
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_nccl_backend, get_torch_device
from veomni.utils.seqlen_pos_transform_utils import pos2culen, prepare_fa_kwargs_from_position_ids

from EfficientQAT.core.quantizer.config import QuantConfig as EQuantConfig
from EfficientQAT.core.quantizer.base_quantizer import BaseQuantizer
from EfficientQAT.core.quantizer.gradual import GradualQuantizer
from EfficientQAT.core.linear.int_quant_linear import (
    convert_linear,
    reinit_quant_params,
    sanitize_quant_params,
    set_quant_state,
)

logger = helper.create_logger(__name__)


class NanDetected(RuntimeError):
    pass


def _tensor_stats(tensor: torch.Tensor, max_non_finite: int = 4) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return stats

    if torch.is_floating_point(tensor):
        with torch.no_grad():
            finite_mask = torch.isfinite(tensor)
            non_finite = (~finite_mask).sum().item()
            stats["non_finite"] = int(non_finite)
            stats["nan"] = int(torch.isnan(tensor).sum().item())
            stats["inf"] = int(torch.isinf(tensor).sum().item())
            if finite_mask.any():
                finite_vals = tensor[finite_mask]
                stats["min"] = finite_vals.min().item()
                stats["max"] = finite_vals.max().item()
                stats["mean"] = finite_vals.mean().item()
                stats["std"] = finite_vals.std().item() if finite_vals.numel() > 1 else 0.0
            else:
                stats["min"] = None
                stats["max"] = None
                stats["mean"] = None
                stats["std"] = None
            if non_finite:
                idx = (~finite_mask).nonzero()
                idx = idx[:max_non_finite]
                stats["non_finite_indices"] = idx.tolist()
                if idx.numel():
                    stats["non_finite_values"] = tensor[tuple(idx.t())].detach().cpu().tolist()
    else:
        stats["min"] = tensor.min().item()
        stats["max"] = tensor.max().item()

    return stats


def _collect_tensor_stats(obj: Any, prefix: str, max_non_finite: int) -> List[Dict[str, Any]]:
    results = []
    for path, tensor in _iter_tensors(obj, prefix):
        results.append({"path": path, "stats": _tensor_stats(tensor, max_non_finite=max_non_finite)})
    return results


def _quantizer_debug_info(q: BaseQuantizer, weight: torch.Tensor, max_non_finite: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "type": q.__class__.__name__,
        "n_bits": int(q.n_bits),
        "group_size": int(q.group_size),
        "enable": bool(q.enable),
        "clamp_method": str(q.clamp_method),
        "input_name": "weight",
    }
    if isinstance(q, GradualQuantizer):
        info["gradual"] = {
            "ratio": float(q.quantization_position_ratio),
            "interpolate_ratio": float(q.interpolate_ratio),
            "has_group_mask": q.group_mask is not None,
            "group_mask_true": int(q.group_mask.sum().item()) if q.group_mask is not None else None,
            "group_mask_len": int(q.group_mask.numel()) if q.group_mask is not None else None,
        }

    steps: Dict[str, Any] = {}
    steps["weight"] = _tensor_stats(weight, max_non_finite=max_non_finite)
    if hasattr(q, "scale"):
        steps["scale_param"] = _tensor_stats(q.scale, max_non_finite=max_non_finite)
    if hasattr(q, "zero_point"):
        steps["zero_point_param"] = _tensor_stats(q.zero_point, max_non_finite=max_non_finite)

    scale_q: Optional[torch.Tensor] = None
    zp_q: Optional[torch.Tensor] = None
    try:
        with torch.no_grad():
            scale_q, zp_q = q.cal_qparams(q.scale, q.zero_point)
        steps["scale_q"] = _tensor_stats(scale_q, max_non_finite=max_non_finite)
        steps["zero_point_q"] = _tensor_stats(zp_q, max_non_finite=max_non_finite)
    except Exception as exc:
        info["qparams_error"] = f"{type(exc).__name__}: {exc}"

    try:
        with torch.no_grad():
            wg = weight.reshape(-1, q.group_size)
            if scale_q is not None:
                w_scaled = wg / scale_q
                steps["w_scaled"] = _tensor_stats(w_scaled, max_non_finite=max_non_finite)
            if scale_q is not None and zp_q is not None:
                w_int = q._quantize(wg, scale_q, zp_q)
                steps["w_int"] = _tensor_stats(w_int, max_non_finite=max_non_finite)
                w_dequant = q._dequantize(w_int, scale_q, zp_q)
                steps["w_dequant"] = _tensor_stats(w_dequant, max_non_finite=max_non_finite)
    except Exception as exc:
        info["quant_debug_error"] = f"{type(exc).__name__}: {exc}"

    info["steps"] = steps
    info["non_finite_steps"] = [
        name for name, s in steps.items() if isinstance(s, dict) and s.get("non_finite", 0) > 0
    ]
    return info


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _build_quant_config(cfg: Dict[str, Any]) -> EQuantConfig:
    cfg = cfg or {}
    base = EQuantConfig()
    kwargs = {}
    for f in fields(EQuantConfig):
        kwargs[f.name] = cfg.get(f.name, getattr(base, f.name))
    return EQuantConfig(**kwargs)


def _iter_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_tensors(v, key)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _iter_tensors(v, key)


def _has_non_finite(tensor: torch.Tensor) -> bool:
    if not torch.is_floating_point(tensor):
        return False
    return not torch.isfinite(tensor).all().item()


class NanTracker:
    def __init__(self, stop_on_first: bool = True, detail: bool = False, max_non_finite: int = 4):
        self.stop_on_first = stop_on_first
        self.detail = detail
        self.max_non_finite = max_non_finite
        self.records: List[Dict[str, Any]] = []

    def record(
        self,
        module_name: str,
        kind: str,
        tensor_path: str,
        tensor: torch.Tensor,
        module: Optional[torch.nn.Module] = None,
        inputs: Any = None,
    ) -> None:
        non_finite = (~torch.isfinite(tensor)).sum().item()
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        record: Dict[str, Any] = {
            "module": module_name,
            "kind": kind,
            "tensor": tensor_path,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "non_finite": int(non_finite),
            "nan": int(nan_count),
            "inf": int(inf_count),
        }
        if module is not None:
            record["module_type"] = module.__class__.__name__
        if self.detail:
            record["tensor_stats"] = _tensor_stats(tensor, max_non_finite=self.max_non_finite)
            if inputs is not None:
                if isinstance(module, BaseQuantizer):
                    if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
                        weight_input = inputs[0]
                    else:
                        weight_input = inputs
                    record["weight_stats"] = _collect_tensor_stats(weight_input, "weight", self.max_non_finite)
                else:
                    record["input_stats"] = _collect_tensor_stats(inputs, "input", self.max_non_finite)
            if isinstance(module, BaseQuantizer):
                try:
                    weight_for_debug = None
                    if hasattr(module, "get_weight_for_priority"):
                        weight_for_debug = module.get_weight_for_priority()
                    if weight_for_debug is None:
                        if isinstance(inputs, (list, tuple)) and inputs:
                            weight_for_debug = inputs[0]
                        else:
                            weight_for_debug = inputs
                    if not torch.is_tensor(weight_for_debug):
                        raise TypeError(f"Quantizer debug expects a tensor, got {type(weight_for_debug)}")
                    record["quantizer_debug"] = _quantizer_debug_info(
                        module, weight_for_debug, self.max_non_finite
                    )
                except Exception as exc:
                    record["quantizer_debug_error"] = f"{type(exc).__name__}: {exc}"
        self.records.append(record)
        if self.stop_on_first:
            raise NanDetected(f"Non-finite detected at {module_name} ({kind}::{tensor_path})")


def _make_hook(name: str, tracker: NanTracker, check_inputs: bool, check_outputs: bool):
    def hook(module, inputs, output):
        if check_inputs:
            for path, tensor in _iter_tensors(inputs, "input"):
                if _has_non_finite(tensor):
                    tracker.record(name, "input", path, tensor, module=module, inputs=inputs)
        if check_outputs:
            for path, tensor in _iter_tensors(output, "output"):
                if _has_non_finite(tensor):
                    tracker.record(name, "output", path, tensor, module=module, inputs=inputs)

    return hook


def _select_modules(model, module_filter: str, filter_mode: str, leaf_only: bool) -> List[Tuple[str, torch.nn.Module]]:
    selected = []
    regex = None
    if module_filter:
        if filter_mode == "regex":
            regex = re.compile(module_filter)
    for name, module in model.named_modules():
        if leaf_only and any(True for _ in module.children()):
            continue
        if module_filter:
            if filter_mode == "prefix":
                if not name.startswith(module_filter):
                    continue
            else:
                if regex is not None and regex.search(name) is None:
                    continue
        selected.append((name, module))
    return selected


def _load_batch(batch_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(batch_path, weights_only=False, map_location="cpu")
    meta: Dict[str, Any] = {}
    if isinstance(payload, dict) and "micro_batch" in payload:
        meta = {k: v for k, v in payload.items() if k != "micro_batch"}
        payload = payload["micro_batch"]
    if isinstance(payload, list):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected batch payload type: {type(payload)}")
    return payload, meta


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _distributed_env_ready() -> bool:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    return all(k in os.environ for k in required)


def _maybe_init_distributed(device: str) -> bool:
    if dist.is_initialized():
        return True
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not _distributed_env_ready():
        logger.warning("WORLD_SIZE=%d but distributed env vars are incomplete; skipping init.", world_size)
        return False
    backend = get_nccl_backend() if device.startswith("cuda") else "gloo"
    dist.init_process_group(backend=backend)
    return True


def _parse_parallel_config(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    def _get_int(key: str, default: int) -> int:
        value = train_cfg.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    return {
        "dp_mode": train_cfg.get("data_parallel_mode", "ddp"),
        "dp_replicate_size": _get_int("data_parallel_replicate_size", -1),
        "dp_shard_size": _get_int("data_parallel_shard_size", -1),
        "tp_size": _get_int("tensor_parallel_size", 1),
        "ep_size": _get_int("expert_parallel_size", 1),
        "pp_size": _get_int("pipeline_parallel_size", 1),
        "ulysses_size": _get_int("ulysses_parallel_size", 1),
        "cp_size": _get_int("context_parallel_size", 1),
        "ep_outside": bool(train_cfg.get("ep_outside", False)),
    }


def _maybe_init_parallel_state(par_cfg: Dict[str, Any]) -> bool:
    if not dist.is_initialized():
        return False
    world_size = dist.get_world_size()
    denom = par_cfg["tp_size"] * par_cfg["pp_size"] * par_cfg["ulysses_size"] * par_cfg["cp_size"]
    if denom <= 0 or world_size % denom != 0:
        logger.warning_rank0(
            "World size %d does not match parallel sizes (tp=%d, pp=%d, ulysses=%d, cp=%d); skipping parallel init.",
            world_size,
            par_cfg["tp_size"],
            par_cfg["pp_size"],
            par_cfg["ulysses_size"],
            par_cfg["cp_size"],
        )
        return False
    dp_size = world_size // denom
    dp_replicate_size = par_cfg["dp_replicate_size"]
    dp_shard_size = par_cfg["dp_shard_size"]
    if dp_replicate_size <= 0 and dp_shard_size <= 0:
        dp_replicate_size = 1
        dp_shard_size = dp_size
    elif dp_replicate_size > 0 and dp_shard_size <= 0:
        if dp_size % dp_replicate_size != 0:
            logger.warning_rank0("Invalid dp_replicate_size=%d for dp_size=%d; using defaults.", dp_replicate_size, dp_size)
            dp_replicate_size = 1
            dp_shard_size = dp_size
        else:
            dp_shard_size = dp_size // dp_replicate_size
    elif dp_shard_size > 0 and dp_replicate_size <= 0:
        if dp_size % dp_shard_size != 0:
            logger.warning_rank0("Invalid dp_shard_size=%d for dp_size=%d; using defaults.", dp_shard_size, dp_size)
            dp_replicate_size = 1
            dp_shard_size = dp_size
        else:
            dp_replicate_size = dp_size // dp_shard_size
    elif dp_replicate_size * dp_shard_size != dp_size:
        logger.warning_rank0(
            "dp_replicate_size * dp_shard_size != dp_size (%d * %d != %d); using defaults.",
            dp_replicate_size,
            dp_shard_size,
            dp_size,
        )
        dp_replicate_size = 1
        dp_shard_size = dp_size

    init_parallel_state(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        tp_size=par_cfg["tp_size"],
        ep_size=par_cfg["ep_size"],
        pp_size=par_cfg["pp_size"],
        cp_size=par_cfg["cp_size"],
        ulysses_size=par_cfg["ulysses_size"],
        dp_mode=par_cfg["dp_mode"],
        device_type=get_device_type(),
        ep_outside=par_cfg["ep_outside"],
    )
    return True


def _infer_sp_rank(meta: Dict[str, Any], sp_size: int) -> int:
    if sp_size <= 1:
        return 0
    for key in ("rank", "global_rank"):
        if key in meta:
            try:
                return int(meta[key]) % sp_size
            except (TypeError, ValueError):
                break
    env_rank = os.getenv("RANK")
    if env_rank is not None:
        try:
            return int(env_rank) % sp_size
        except ValueError:
            return 0
    return 0


def _slice_sequence_tensor(tensor: torch.Tensor, sp_size: int, sp_rank: int) -> torch.Tensor:
    seq_len = tensor.shape[-1]
    sp_chunk = (seq_len + sp_size - 1) // sp_size
    start = sp_rank * sp_chunk
    end = min(start + sp_chunk, seq_len)
    return tensor[..., start:end].contiguous()


def _refresh_flash_attention_kwargs(batch: Dict[str, Any]) -> None:
    if "position_ids" not in batch:
        return
    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
        batch["position_ids"]
    )
    batch["cu_seq_lens_q"] = cu_seq_lens_q
    batch["cu_seq_lens_k"] = cu_seq_lens_k
    batch["max_length_q"] = max_length_q
    batch["max_length_k"] = max_length_k
    if "cu_seqlens" in batch:
        batch["cu_seqlens"] = pos2culen(batch["position_ids"])


def _maybe_slice_sp_batch(batch: Dict[str, Any], sp_size: int, sp_rank: int) -> bool:
    if sp_size <= 1 or "input_ids" not in batch or not torch.is_tensor(batch["input_ids"]):
        return False
    target_len = batch["input_ids"].shape[-1]
    changed = False
    for key in ("position_ids", "attention_mask", "labels"):
        tensor = batch.get(key)
        if not torch.is_tensor(tensor):
            continue
        if tensor.shape[-1] == target_len:
            continue
        if tensor.shape[-1] < target_len:
            logger.warning_rank0("Skip slicing %s (len=%d < input_ids len=%d).", key, tensor.shape[-1], target_len)
            continue
        batch[key] = _slice_sequence_tensor(tensor, sp_size, sp_rank)
        changed = True
    if changed:
        _refresh_flash_attention_kwargs(batch)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe NaN/Inf in module inputs/outputs.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument("--checkpoint-path", default="", help="Path to DCP/bytecheckpoint/omnistore checkpoint.")
    parser.add_argument("--ckpt-manager", default="dcp", help="Checkpoint manager: dcp|omnistore|bytecheckpoint.")
    parser.add_argument("--weights-path", default="", help="Optional HF weights path (if no checkpoint).")
    parser.add_argument("--batch-path", required=True, help="Path to saved micro batch.")
    parser.add_argument("--module-filter", default="", help="Module name filter (regex or prefix).")
    parser.add_argument("--filter-mode", choices=["regex", "prefix"], default="regex")
    parser.add_argument("--all-modules", action="store_true", help="Check all modules (not only leaf modules).")
    parser.add_argument("--check-inputs", action="store_true", default=True, help="Check module inputs.")
    parser.add_argument("--no-check-inputs", action="store_true", help="Disable input checks.")
    parser.add_argument("--check-outputs", action="store_true", default=True, help="Check module outputs.")
    parser.add_argument("--no-check-outputs", action="store_true", help="Disable output checks.")
    parser.add_argument("--device", default="cuda", help="Device to run the probe on.")
    parser.add_argument("--report-path", default="", help="Optional JSON report path.")
    parser.add_argument("--stop-on-first", action="store_true", default=True, help="Stop at first NaN/Inf.")
    parser.add_argument("--no-stop-on-first", action="store_true", help="Continue after detections.")
    parser.add_argument("--detail", action="store_true", help="Collect detailed stats for the first non-finite.")
    parser.add_argument(
        "--max-non-finite",
        type=int,
        default=4,
        help="Max number of non-finite indices/values recorded per tensor in detailed mode.",
    )
    args = parser.parse_args()

    if args.no_check_inputs:
        check_inputs = False
    else:
        check_inputs = True
    if args.no_check_outputs:
        check_outputs = False
    else:
        check_outputs = True
    stop_on_first = args.stop_on_first and not args.no_stop_on_first
    leaf_only = not args.all_modules

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    quant_cfg = cfg.get("quantizer", {})
    parallel_cfg = _parse_parallel_config(train_cfg)

    config_path = model_cfg.get("config_path") or model_cfg.get("model_path")
    if config_path is None:
        raise ValueError("config_path/model_path not found in config.")

    weights_path = args.weights_path or model_cfg.get("model_path")
    if args.checkpoint_path:
        weights_path = None

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning_rank0("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    if _maybe_init_distributed(args.device):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if args.device.startswith("cuda"):
            get_torch_device().set_device(f"{get_device_type()}:{local_rank}")
            args.device = f"cuda:{local_rank}"

    parallel_initialized = _maybe_init_parallel_state(parallel_cfg)

    torch_dtype = "bfloat16"
    model = build_foundation_model(
        config_path=config_path,
        weights_path=weights_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        moe_implementation=model_cfg.get("moe_implementation"),
        init_device="cuda" if args.device.startswith("cuda") else "cpu",
        force_use_huggingface=model_cfg.get("force_use_huggingface", False),
    )

    qcfg = _build_quant_config(quant_cfg)
    convert_linear(model, prefix="", config=qcfg)

    if args.checkpoint_path:
        logger.info_rank0("Loading checkpoint state_dict from %s", args.checkpoint_path)
        state_dict = ckpt_to_state_dict(args.checkpoint_path, output_dir=args.checkpoint_path, ckpt_manager=args.ckpt_manager)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning_rank0("Missing keys: %d", len(missing))
        if unexpected:
            logger.warning_rank0("Unexpected keys: %d", len(unexpected))
    else:
        reinit_quant_params(model)
        sanitize_quant_params(model)

    set_quant_state(model, weight_quant=True)
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    batch, batch_meta = _load_batch(args.batch_path)
    if not parallel_initialized:
        sp_size = parallel_cfg["ulysses_size"] * parallel_cfg["cp_size"]
        sp_rank = _infer_sp_rank(batch_meta, sp_size)
        if _maybe_slice_sp_batch(batch, sp_size=sp_size, sp_rank=sp_rank):
            logger.warning_rank0(
                "Sequence-parallel batch detected without initialized parallel state; sliced inputs for sp_rank=%d.",
                sp_rank,
            )
    batch = _move_batch(batch, device=device)

    tracker = NanTracker(stop_on_first=stop_on_first, detail=args.detail, max_non_finite=args.max_non_finite)
    hooks = []
    for name, module in _select_modules(model, args.module_filter, args.filter_mode, leaf_only):
        hooks.append(module.register_forward_hook(_make_hook(name, tracker, check_inputs, check_outputs)))
    logger.info_rank0("Registered %d forward hooks.", len(hooks))

    try:
        try:
            with torch.no_grad():
                model(**batch, use_cache=False)
        except TypeError:
            with torch.no_grad():
                model(**batch)
    except NanDetected as exc:
        logger.warning_rank0("NaN/Inf detected: %s", exc)
    finally:
        for h in hooks:
            h.remove()

    if tracker.records:
        logger.info_rank0("Found %d non-finite records.", len(tracker.records))
    else:
        logger.info_rank0("No non-finite tensors detected.")

    if args.report_path:
        report_dir = os.path.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(tracker.records, f, indent=2)
        logger.info_rank0("Saved report to %s", args.report_path)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
