from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import os
import re

import torch
import torch.nn as nn


@dataclass
class ModuleCallEvent:
    name: str
    type: str
    call_idx: int
    input_spec: Any
    output_spec: Any
    output_ref: Any | None = None

    @property
    def key(self) -> str:
        return f"{self.name}#{self.call_idx}"


@dataclass
class NodeDiff:
    key: str
    name: str
    call_idx: int
    type_a: str | None
    type_b: str | None
    metrics: Dict[str, float]
    tensor_count: int
    missing_paths: List[str] = field(default_factory=list)
    per_position_rel_l2: List[float] | None = None


@dataclass
class CompareReport:
    nodes: List[NodeDiff]
    missing_in_a: List[str]
    missing_in_b: List[str]
    first_diverge: NodeDiff | None
    logits: Dict[str, Any] | None


@dataclass
class BreakpointReport:
    strategy: str
    baseline_distance: float
    found_index: int | None
    found_key: str | None
    history: List[Dict[str, Any]]


@dataclass
class TwinCompareConfig:
    include_types: Tuple[type, ...] = (nn.Linear,)
    include_name_regex: str | None = None
    exclude_name_regex: str | None = None
    save_outputs: bool = False
    save_outputs_on_types: Tuple[type, ...] = (nn.Linear,)
    metrics: Tuple[str, ...] = ("rel_l2", "cos", "max_abs", "mean_abs")
    per_position: bool = True
    logits_compare: bool = True
    mode: str = "decoupled"
    stop_on_first_diverge: bool = False
    diverge_threshold: float = 0.05
    topk: int = 5
    dump_dir: str | None = None
    summary_stats: bool = False
    breakpoint_threshold: float = 0.02


def record_graph(
    model: nn.Module,
    example_inputs: Any,
    cfg: TwinCompareConfig,
    example_kwargs: Optional[Dict[str, Any]] = None,
) -> List[ModuleCallEvent]:
    recorder = GraphRecorder(model, cfg, save_outputs_override=cfg.save_outputs)
    _run_with_hooks(model, recorder, example_inputs, example_kwargs)
    return recorder.events


def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    example_inputs: Any,
    cfg: Optional[TwinCompareConfig] = None,
    example_kwargs: Optional[Dict[str, Any]] = None,
) -> CompareReport:
    cfg = cfg or TwinCompareConfig()
    example_kwargs = example_kwargs or {}

    model_a.eval()
    model_b.eval()

    recorder_a = GraphRecorder(model_a, cfg, save_outputs_override=True)
    out_a = _run_with_hooks(model_a, recorder_a, example_inputs, example_kwargs)

    if cfg.mode == "decoupled":
        recorder_b = GraphRecorder(model_b, cfg, save_outputs_override=True)
    elif cfg.mode == "coupled":
        recorder_b = GraphRecorder(
            model_b,
            cfg,
            save_outputs_override=True,
            output_override=_make_injector(recorder_a.outputs),
        )
    else:
        raise ValueError(f"Unsupported mode {cfg.mode}")

    out_b = _run_with_hooks(model_b, recorder_b, example_inputs, example_kwargs)

    nodes, missing_in_a, missing_in_b, first_diverge = _compare_recorders(
        recorder_a, recorder_b, cfg
    )

    logits_report = None
    if cfg.logits_compare:
        logits_a = _extract_logits(out_a)
        logits_b = _extract_logits(out_b)
        if logits_a is not None and logits_b is not None:
            logits_report = compare_logits_per_position(logits_a, logits_b, topk=cfg.topk)

    report = CompareReport(
        nodes=nodes,
        missing_in_a=missing_in_a,
        missing_in_b=missing_in_b,
        first_diverge=first_diverge,
        logits=logits_report,
    )
    if cfg.dump_dir:
        _dump_compare_report(cfg.dump_dir, report)
    return report


def find_irreversible_breakpoint(
    model_a: nn.Module,
    model_b: nn.Module,
    example_inputs: Any,
    cfg: Optional[TwinCompareConfig] = None,
    example_kwargs: Optional[Dict[str, Any]] = None,
    strategy: str = "bisect",
) -> BreakpointReport:
    cfg = cfg or TwinCompareConfig()
    cfg = TwinCompareConfig(**{**cfg.__dict__, "save_outputs": True})
    example_kwargs = example_kwargs or {}

    model_a.eval()
    model_b.eval()

    recorder_a = GraphRecorder(model_a, cfg, save_outputs_override=True)
    out_a = _run_with_hooks(model_a, recorder_a, example_inputs, example_kwargs)
    logits_a = _extract_logits(out_a)

    if logits_a is None:
        raise RuntimeError("Model A did not return logits for breakpoint search.")

    keys = [e.key for e in recorder_a.events if e.key in recorder_a.outputs]

    baseline = _run_with_injection(
        model_b, cfg, example_inputs, example_kwargs, inject_map=None
    )
    baseline_logits = _extract_logits(baseline)
    baseline_distance = _logits_distance(logits_a, baseline_logits)

    history: List[Dict[str, Any]] = []
    found_index: int | None = None
    found_key: str | None = None

    if not keys:
        return BreakpointReport(
            strategy=strategy,
            baseline_distance=baseline_distance,
            found_index=None,
            found_key=None,
            history=history,
        )

    def run_prefix(idx: int) -> float:
        inject_keys = set(keys[: idx + 1])
        inject_map = {k: recorder_a.outputs[k] for k in inject_keys}
        out_b = _run_with_injection(model_b, cfg, example_inputs, example_kwargs, inject_map)
        logits_b = _extract_logits(out_b)
        dist = _logits_distance(logits_a, logits_b)
        history.append({"index": idx, "key": keys[idx], "distance": dist})
        return dist

    if strategy == "linear":
        for idx in range(len(keys)):
            dist = run_prefix(idx)
            if dist <= cfg.breakpoint_threshold:
                found_index = idx
                found_key = keys[idx]
                break
    elif strategy == "bisect":
        lo = 0
        hi = len(keys) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            dist = run_prefix(mid)
            if dist <= cfg.breakpoint_threshold:
                found_index = mid
                found_key = keys[mid]
                hi = mid - 1
            else:
                lo = mid + 1
    else:
        raise ValueError(f"Unsupported strategy {strategy}")

    report = BreakpointReport(
        strategy=strategy,
        baseline_distance=baseline_distance,
        found_index=found_index,
        found_key=found_key,
        history=history,
    )
    if cfg.dump_dir:
        _dump_breakpoint_report(cfg.dump_dir, report)
    return report


def compare_logits_per_position(
    logits_a: torch.Tensor, logits_b: torch.Tensor, topk: int = 5
) -> Dict[str, Any]:
    if logits_a.shape != logits_b.shape:
        return {"error": f"shape mismatch: a={tuple(logits_a.shape)} b={tuple(logits_b.shape)}"}

    la = logits_a.float()
    lb = logits_b.float()
    if la.dim() == 2:
        la = la.unsqueeze(1)
        lb = lb.unsqueeze(1)

    top1_a = la.argmax(dim=-1)
    top1_b = lb.argmax(dim=-1)
    top1_agree = (top1_a == top1_b).float().mean(dim=0)

    k = min(topk, la.shape[-1])
    topk_a = la.topk(k=k, dim=-1).indices
    topk_b = lb.topk(k=k, dim=-1).indices
    eq = topk_a.unsqueeze(-1) == topk_b.unsqueeze(-2)
    overlap = eq.any(dim=-1).sum(dim=-1).float() / float(k)
    topk_overlap = overlap.mean(dim=0)

    log_p_a = torch.log_softmax(la, dim=-1)
    log_p_b = torch.log_softmax(lb, dim=-1)
    p_a = log_p_a.exp()
    kl = (p_a * (log_p_a - log_p_b)).sum(dim=-1)
    kl_mean = kl.mean(dim=0)

    margin_diff = None
    if la.shape[-1] >= 2:
        top2_a = la.topk(k=2, dim=-1).values
        top2_b = lb.topk(k=2, dim=-1).values
        margin_a = top2_a[..., 0] - top2_a[..., 1]
        margin_b = top2_b[..., 0] - top2_b[..., 1]
        margin_diff = (margin_a - margin_b).mean(dim=0)

    diff_metrics = _tensor_diff_metrics(la, lb, metrics=("rel_l2", "max_abs", "mean_abs"))

    return {
        "rel_l2": diff_metrics.get("rel_l2"),
        "max_abs": diff_metrics.get("max_abs"),
        "mean_abs": diff_metrics.get("mean_abs"),
        "top1_agree": top1_agree.tolist(),
        "topk_overlap": topk_overlap.tolist(),
        "kl": kl_mean.tolist(),
        "margin_diff": None if margin_diff is None else margin_diff.tolist(),
    }


def print_compare_report(report: CompareReport, topk: int = 10, metric: str = "rel_l2") -> None:
    print(
        f"aligned nodes: {len(report.nodes)} "
        f"missing_in_a={len(report.missing_in_a)} missing_in_b={len(report.missing_in_b)}"
    )
    if report.first_diverge:
        val = report.first_diverge.metrics.get(metric)
        print(
            f"first diverge: {report.first_diverge.key} {metric}={val:.4g}"
            if val is not None
            else f"first diverge: {report.first_diverge.key}"
        )

    if report.missing_in_a:
        print(f"missing in A (first 10): {report.missing_in_a[:10]}")
    if report.missing_in_b:
        print(f"missing in B (first 10): {report.missing_in_b[:10]}")

    nodes = sorted(report.nodes, key=lambda n: n.metrics.get(metric, 0.0), reverse=True)
    print(f"top {min(topk, len(nodes))} mismatches by {metric}:")
    for node in nodes[:topk]:
        val = node.metrics.get(metric)
        val_str = "na" if val is None else f"{val:.4g}"
        print(f"{node.key}: {metric}={val_str} type_a={node.type_a} type_b={node.type_b}")

    if report.logits:
        logits_rel = report.logits.get("rel_l2")
        if logits_rel is not None:
            print(f"logits rel_l2={logits_rel:.4g}")


class GraphRecorder:
    def __init__(
        self,
        model: nn.Module,
        cfg: TwinCompareConfig,
        save_outputs_override: Optional[bool] = None,
        output_override: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.events: List[ModuleCallEvent] = []
        self.outputs: Dict[str, Any] = {}
        self._handles: List[Any] = []
        self._call_counter: Dict[str, int] = defaultdict(int)
        self._save_outputs_override = save_outputs_override
        self._output_override = output_override

    def __enter__(self) -> "GraphRecorder":
        for name, module in _iter_hook_modules(self.model, self.cfg):
            handle = module.register_forward_hook(self._make_hook(name, module))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str, module: nn.Module):
        def hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any):
            call_idx = self._call_counter[name]
            self._call_counter[name] += 1
            key = f"{name}#{call_idx}"

            input_spec = _summarize_obj(inputs, self.cfg.summary_stats)
            output_spec = _summarize_obj(output, self.cfg.summary_stats)

            out_for_record = output
            output_ref = None
            if self._should_save_output(module):
                output_ref = _detach_obj(out_for_record)
                self.outputs[key] = output_ref

            event = ModuleCallEvent(
                name=name,
                type=module.__class__.__name__,
                call_idx=call_idx,
                input_spec=input_spec,
                output_spec=output_spec,
                output_ref=output_ref,
            )
            self.events.append(event)

            if self._output_override is not None:
                override = self._output_override(key, module, output)
                if override is not None:
                    return override
            return None

        return hook

    def _should_save_output(self, module: nn.Module) -> bool:
        if self._save_outputs_override is not None:
            return self._save_outputs_override
        if self.cfg.save_outputs:
            return True
        if self.cfg.save_outputs_on_types and isinstance(module, self.cfg.save_outputs_on_types):
            return True
        return False


def _iter_hook_modules(model: nn.Module, cfg: TwinCompareConfig) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if _should_hook_module(name, module, cfg):
            yield name, module


def _should_hook_module(name: str, module: nn.Module, cfg: TwinCompareConfig) -> bool:
    if cfg.include_types:
        if not isinstance(module, cfg.include_types):
            return False
    else:
        if len(list(module.children())) > 0:
            return False

    if cfg.include_name_regex and not re.search(cfg.include_name_regex, name):
        return False
    if cfg.exclude_name_regex and re.search(cfg.exclude_name_regex, name):
        return False
    return True


def _make_injector(output_map: Dict[str, Any]):
    def injector(key: str, module: nn.Module, output: Any) -> Any | None:
        if key not in output_map:
            return None
        target = output_map[key]
        device = _infer_device(output)
        if device is not None:
            target = _to_device(target, device)
        return target

    return injector


def _run_with_hooks(
    model: nn.Module,
    recorder: GraphRecorder,
    example_inputs: Any,
    example_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    inputs = _normalize_inputs(example_inputs)
    with recorder:
        with torch.no_grad():
            return model(*inputs, **(example_kwargs or {}))


def _normalize_inputs(example_inputs: Any) -> Tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _compare_recorders(
    recorder_a: GraphRecorder, recorder_b: GraphRecorder, cfg: TwinCompareConfig
) -> Tuple[List[NodeDiff], List[str], List[str], NodeDiff | None]:
    events_a = {e.key: e for e in recorder_a.events}
    events_b = {e.key: e for e in recorder_b.events}
    keys_a = [e.key for e in recorder_a.events]
    keys_b = [e.key for e in recorder_b.events]

    set_a = set(keys_a)
    set_b = set(keys_b)
    missing_in_b = sorted(set_a - set_b)
    missing_in_a = sorted(set_b - set_a)

    nodes: List[NodeDiff] = []
    first_diverge = None

    for key in keys_a:
        if key not in events_b:
            continue
        out_a = recorder_a.outputs.get(key)
        out_b = recorder_b.outputs.get(key)
        if out_a is None or out_b is None:
            continue
        metrics, tensor_count, missing_paths, per_position = _compare_outputs(out_a, out_b, cfg)
        node = NodeDiff(
            key=key,
            name=events_a[key].name,
            call_idx=events_a[key].call_idx,
            type_a=events_a[key].type,
            type_b=events_b[key].type,
            metrics=metrics,
            tensor_count=tensor_count,
            missing_paths=missing_paths,
            per_position_rel_l2=per_position,
        )
        nodes.append(node)
        if (
            first_diverge is None
            and cfg.diverge_threshold is not None
            and metrics.get("rel_l2") is not None
            and metrics["rel_l2"] > cfg.diverge_threshold
        ):
            first_diverge = node
            if cfg.stop_on_first_diverge:
                break

    return nodes, missing_in_a, missing_in_b, first_diverge


def _compare_outputs(
    out_a: Any, out_b: Any, cfg: TwinCompareConfig
) -> Tuple[Dict[str, float], int, List[str], List[float] | None]:
    a_map = dict(_flatten_tensors(out_a))
    b_map = dict(_flatten_tensors(out_b))
    common = sorted(set(a_map) & set(b_map))
    missing_paths = sorted((set(a_map) | set(b_map)) - set(common))

    per_tensor_metrics: List[Dict[str, Any]] = []
    for path in common:
        ta = a_map[path]
        tb = b_map[path]
        if ta.shape != tb.shape:
            continue
        metrics = _tensor_diff_metrics(ta, tb, cfg.metrics)
        per_tensor_metrics.append(metrics)

    metrics = _aggregate_metrics(per_tensor_metrics, cfg.metrics)
    per_position = None
    if cfg.per_position and common:
        ta = a_map[common[0]]
        tb = b_map[common[0]]
        if ta.shape == tb.shape:
            per_position = _per_position_rel_l2(ta, tb)

    return metrics, len(per_tensor_metrics), missing_paths, per_position


def _tensor_diff_metrics(
    a: torch.Tensor, b: torch.Tensor, metrics: Sequence[str]
) -> Dict[str, float]:
    a_f = a.float()
    b_f = b.float()
    diff = a_f - b_f
    eps = 1e-8
    out: Dict[str, float] = {}

    if "max_abs" in metrics:
        out["max_abs"] = diff.abs().max().item()
    if "mean_abs" in metrics:
        out["mean_abs"] = diff.abs().mean().item()
    if "rel_l2" in metrics:
        out["rel_l2"] = diff.norm().item() / (a_f.norm().item() + eps)
    if "cos" in metrics:
        denom = (a_f.norm().item() * b_f.norm().item()) + eps
        out["cos"] = (a_f.flatten() @ b_f.flatten()).item() / denom
    if "sign_mismatch" in metrics:
        out["sign_mismatch"] = (a_f.sign() != b_f.sign()).float().mean().item()
    if "nan_a" in metrics:
        out["nan_a"] = float(torch.isnan(a_f).any().item())
    if "nan_b" in metrics:
        out["nan_b"] = float(torch.isnan(b_f).any().item())
    if "inf_a" in metrics:
        out["inf_a"] = float(torch.isinf(a_f).any().item())
    if "inf_b" in metrics:
        out["inf_b"] = float(torch.isinf(b_f).any().item())
    return out


def _aggregate_metrics(
    per_tensor: List[Dict[str, float]], metrics: Sequence[str]
) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not per_tensor:
        return agg
    for name in metrics:
        vals = [m[name] for m in per_tensor if name in m]
        if not vals:
            continue
        if name == "max_abs":
            agg[name] = max(vals)
        else:
            agg[name] = sum(vals) / len(vals)
    return agg


def _per_position_rel_l2(a: torch.Tensor, b: torch.Tensor) -> List[float] | None:
    if a.dim() < 2:
        return None
    a_f = a.float()
    b_f = b.float()
    diff = a_f - b_f
    a2 = a_f.reshape(a_f.shape[0], a_f.shape[1], -1)
    d2 = diff.reshape(diff.shape[0], diff.shape[1], -1)
    eps = 1e-8
    rel = d2.norm(dim=-1) / (a2.norm(dim=-1) + eps)
    return rel.mean(dim=0).tolist()


def _flatten_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield (prefix or "output"), obj
        return
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _flatten_tensors(item, child)
        return
    if isinstance(obj, dict):
        for key, item in obj.items():
            key_str = str(key)
            child = f"{prefix}.{key_str}" if prefix else key_str
            yield from _flatten_tensors(item, child)
        return


def _summarize_obj(obj: Any, with_stats: bool) -> Any:
    if torch.is_tensor(obj):
        return _summarize_tensor(obj, with_stats)
    if isinstance(obj, (list, tuple)):
        return [_summarize_obj(item, with_stats) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _summarize_obj(v, with_stats) for k, v in obj.items()}
    return {"type": type(obj).__name__}


def _summarize_tensor(tensor: torch.Tensor, with_stats: bool) -> Dict[str, Any]:
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }
    if not with_stats:
        return summary
    t = tensor.float()
    summary.update(
        {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "max_abs": t.abs().max().item(),
            "nan": int(torch.isnan(t).any().item()),
            "inf": int(torch.isinf(t).any().item()),
        }
    )
    return summary


def _detach_obj(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, list):
        return [_detach_obj(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_obj(item) for item in obj)
    if isinstance(obj, dict):
        return {k: _detach_obj(v) for k, v in obj.items()}
    return obj


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device=device)
    if isinstance(obj, list):
        return [_to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _infer_device(obj: Any) -> Optional[torch.device]:
    tensor = _find_first_tensor(obj)
    if tensor is None:
        return None
    return tensor.device


def _find_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(obj, dict):
        for item in obj.values():
            found = _find_first_tensor(item)
            if found is not None:
                return found
        return None
    return None


def _extract_logits(output: Any) -> Optional[torch.Tensor]:
    if output is None:
        return None
    if torch.is_tensor(output):
        return output
    if hasattr(output, "logits"):
        logits = getattr(output, "logits")
        if torch.is_tensor(logits):
            return logits
    if isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    return None


def _run_with_injection(
    model: nn.Module,
    cfg: TwinCompareConfig,
    example_inputs: Any,
    example_kwargs: Optional[Dict[str, Any]],
    inject_map: Optional[Dict[str, Any]],
) -> Any:
    if not inject_map:
        with torch.no_grad():
            return model(*_normalize_inputs(example_inputs), **(example_kwargs or {}))

    def override(key: str, module: nn.Module, output: Any) -> Any | None:
        if key not in inject_map:
            return None
        target = inject_map[key]
        device = _infer_device(output)
        if device is not None:
            target = _to_device(target, device)
        return target

    recorder = GraphRecorder(model, cfg, save_outputs_override=False, output_override=override)
    return _run_with_hooks(model, recorder, example_inputs, example_kwargs)


def _logits_distance(a: torch.Tensor, b: Optional[torch.Tensor]) -> float:
    if b is None:
        return float("inf")
    metrics = _tensor_diff_metrics(a, b, metrics=("rel_l2",))
    return metrics.get("rel_l2", float("inf"))


def _dump_compare_report(path: str, report: CompareReport) -> None:
    os.makedirs(path, exist_ok=True)
    payload = {
        "nodes": [
            {
                "key": n.key,
                "name": n.name,
                "call_idx": n.call_idx,
                "type_a": n.type_a,
                "type_b": n.type_b,
                "metrics": n.metrics,
                "tensor_count": n.tensor_count,
                "missing_paths": n.missing_paths,
                "per_position_rel_l2": n.per_position_rel_l2,
            }
            for n in report.nodes
        ],
        "missing_in_a": report.missing_in_a,
        "missing_in_b": report.missing_in_b,
        "first_diverge": None
        if report.first_diverge is None
        else {
            "key": report.first_diverge.key,
            "metrics": report.first_diverge.metrics,
        },
        "logits": report.logits,
    }
    report_path = os.path.join(path, "twin_compare_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _dump_breakpoint_report(path: str, report: BreakpointReport) -> None:
    os.makedirs(path, exist_ok=True)
    payload = {
        "strategy": report.strategy,
        "baseline_distance": report.baseline_distance,
        "found_index": report.found_index,
        "found_key": report.found_key,
        "history": report.history,
    }
    report_path = os.path.join(path, "twin_breakpoint_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
