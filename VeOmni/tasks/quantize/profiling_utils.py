import math
import re
import re
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from veomni.utils.helper import EnvironMeter
from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear


class MetricProbe:
    def bind(self, model: torch.nn.Module) -> None:
        pass

    def collect(self, model: torch.nn.Module) -> Dict[str, float]:
        return {}


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
        if hasattr(tensor, "wait"):
            tensor = tensor.wait()
    return tensor


def _matches_name(name: str, patterns: Sequence[str]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)


def _matches_name_or_prefix(name: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if name == pattern or name.endswith(f".{pattern}"):
            return True
        if name.startswith(f"{pattern}."):
            return True
    return False


def _resolve_modules(
    model: torch.nn.Module, module_names: Sequence[str]
) -> List[Tuple[str, torch.nn.Module]]:
    if not module_names:
        return []
    named = dict(model.named_modules())
    resolved = []
    for name in module_names:
        if name in named:
            resolved.append((name, named[name]))
    if resolved:
        return resolved
    for name, module in named.items():
        if _matches_name(name, module_names):
            resolved.append((name, module))
    return resolved


def _resolve_linear_modules(
    model: torch.nn.Module, module_names: Sequence[str]
) -> List[Tuple[str, torch.nn.Module]]:
    if not module_names:
        return []
    resolved: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and _matches_name_or_prefix(name, module_names):
            resolved.append((name, module))
    return resolved


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    found = False
    for param in parameters:
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        param_norm = grad.float().norm(2).item()
        total_sq += param_norm ** 2
        found = True
    return math.sqrt(total_sq) if found else 0.0


def _topk_snapshot(values: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if values.numel() == 0:
        return values, values
    k = max(min(int(topk), values.numel()), 1)
    topk_vals, topk_idx = torch.topk(values, k)
    return topk_vals.detach().cpu(), topk_idx.detach().cpu()


class GradNormProbe(MetricProbe):
    def __init__(self, module_names: Sequence[str], metric_prefix: str = "grad_norm"):
        self.module_names = list(module_names)
        self.metric_prefix = metric_prefix
        self._targets: List[Tuple[str, torch.nn.Module]] = []

    def bind(self, model: torch.nn.Module) -> None:
        self._targets = _resolve_modules(model, self.module_names)

    def collect(self, model: torch.nn.Module) -> Dict[str, float]:
        if not self._targets:
            self.bind(model)
        metrics: Dict[str, float] = {}
        for name, module in self._targets:
            metrics[f"{self.metric_prefix}/{name}"] = _grad_norm(module.parameters())
        return metrics


class LinearGradStatsProbe(MetricProbe):
    def __init__(
        self,
        module_names: Sequence[str],
        metric_prefix: str = "monitor/linear_grad",
        interval: int = 1,
        snapshot_interval: int = 0,
        snapshot_steps: Optional[Sequence[int]] = None,
        snapshot_histogram: bool = False,
        snapshot_histogram_bins: int = 64,
        snapshot_topk_bar: bool = False,
        snapshot_histogram_group_by: str = "module",
        snapshot_histogram_sample: int = 0,
        snapshot_topk_aggregate: bool = False,
        topk: int = 20,
        spectral_iters: int = 2,
        sign_flip_sample: int = 65536,
        eps: float = 1e-12,
    ) -> None:
        self.module_names = list(module_names)
        self.metric_prefix = metric_prefix
        self.interval = int(interval)
        self.snapshot_interval = int(snapshot_interval)
        self.snapshot_steps = {int(s) for s in (snapshot_steps or []) if int(s) > 0}
        self.snapshot_histogram = bool(snapshot_histogram)
        self.snapshot_histogram_bins = max(int(snapshot_histogram_bins), 1)
        self.snapshot_topk_bar = bool(snapshot_topk_bar)
        self.snapshot_histogram_group_by = snapshot_histogram_group_by
        self.snapshot_histogram_sample = max(int(snapshot_histogram_sample), 0)
        self.snapshot_topk_aggregate = bool(snapshot_topk_aggregate)
        self.topk = int(topk)
        self.spectral_iters = int(spectral_iters)
        self.sign_flip_sample = int(sign_flip_sample)
        self.eps = float(eps)
        self._targets: List[Tuple[str, torch.nn.Module]] = []
        self._prev_sign: Dict[str, torch.Tensor] = {}
        self._sign_indices: Dict[str, torch.Tensor] = {}
        self._step: Optional[int] = None

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def _get_wandb(self):
        if not (self.snapshot_histogram or self.snapshot_topk_bar or self.snapshot_topk_aggregate):
            return None
        try:
            import wandb  # type: ignore
        except Exception:
            return None
        if getattr(wandb, "run", None) is None:
            return None
        return wandb

    def _hist_group_key(self, name: str) -> str:
        group_by = (self.snapshot_histogram_group_by or "module").lower()
        if group_by == "module":
            return name
        if group_by == "layer":
            match = re.search(r"(?:^|\\.)(?:layers|layer)\\.(\\d+)(?:\\.|$)", name)
            if match:
                return f"layer_{match.group(1)}"
            return "layer_unknown"
        if group_by == "suffix":
            match = re.search(r"(?:^|\\.)(?:layers|layer)\\.\\d+\\.(.+)$", name)
            if match:
                return match.group(1)
            return name.split(".")[-1]
        return name

    def _maybe_sample(self, values: torch.Tensor) -> torch.Tensor:
        if self.snapshot_histogram_sample <= 0:
            return values
        flat = values.reshape(-1)
        if flat.numel() <= self.snapshot_histogram_sample:
            return flat
        idx = torch.randperm(flat.numel(), device=flat.device)[: self.snapshot_histogram_sample]
        return flat[idx]

    def bind(self, model: torch.nn.Module) -> None:
        self._targets = _resolve_linear_modules(model, self.module_names)

    def _sample_sign(self, name: str, flat: torch.Tensor) -> torch.Tensor:
        numel = flat.numel()
        if self.sign_flip_sample <= 0 or self.sign_flip_sample >= numel:
            return torch.sign(flat)
        indices = self._sign_indices.get(name)
        if indices is None or indices.numel() != self.sign_flip_sample or indices.device != flat.device:
            indices = torch.randperm(numel, device=flat.device)[: self.sign_flip_sample]
            self._sign_indices[name] = indices
        return torch.sign(flat[indices])

    def _sign_flip_rate(self, name: str, grad: torch.Tensor) -> float:
        flat = grad.reshape(-1)
        if flat.numel() == 0:
            return 0.0
        sign = self._sample_sign(name, flat)
        store_on_cpu = self.sign_flip_sample > 0 and self.sign_flip_sample < flat.numel()
        if store_on_cpu:
            sign = sign.to(dtype=torch.int8).cpu()
        else:
            sign = sign.to(dtype=torch.int8)
        prev = self._prev_sign.get(name)
        self._prev_sign[name] = sign
        if prev is None or prev.numel() != sign.numel():
            return 0.0
        if prev.device != sign.device:
            prev = prev.to(sign.device)
        valid = (sign != 0) & (prev != 0)
        if not bool(valid.any()):
            return 0.0
        flips = (sign != prev) & valid
        return float(flips.sum().item() / (valid.sum().item() + self.eps))

    def collect(self, model: torch.nn.Module) -> Dict[str, float]:
        if not self._targets:
            self.bind(model)
        if not self._targets or self.interval <= 0:
            return {}
        step = self._step or 0
        if step % self.interval != 0:
            return {}
        if self.snapshot_steps:
            snapshot = step in self.snapshot_steps
        else:
            snapshot = self.snapshot_interval > 0 and step % self.snapshot_interval == 0
        wandb = self._get_wandb() if snapshot else None
        hist_accum: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        topk_accum: Dict[str, List[torch.Tensor]] = {"row": [], "col": []}
        metrics: Dict[str, float] = {}
        for name, module in self._targets:
            weight = getattr(module, "weight", None)
            if weight is None or weight.grad is None:
                continue
            grad = _to_local_tensor(weight.grad.detach())
            if grad.is_sparse:
                grad = grad.coalesce().values()
            grad = grad.float()

            grad_fro = grad.norm().item()
            weight_fro = _to_local_tensor(weight.detach()).float().norm().item()
            metrics[f"{self.metric_prefix}/{name}/grad_fro"] = grad_fro
            metrics[f"{self.metric_prefix}/{name}/update_ratio"] = grad_fro / (weight_fro + self.eps)

            if grad.ndim >= 2:
                row_energy = grad.pow(2).sum(dim=1)
                col_energy = grad.pow(2).sum(dim=0)
                if snapshot:
                    row_vals = None
                    col_vals = None
                    if self.snapshot_topk_aggregate or self.snapshot_topk_bar:
                        row_vals, row_idx = _topk_snapshot(row_energy, self.topk)
                        col_vals, col_idx = _topk_snapshot(col_energy, self.topk)
                        if self.snapshot_topk_aggregate:
                            topk_accum["row"].append(row_vals.detach().float().cpu())
                            topk_accum["col"].append(col_vals.detach().float().cpu())
                    if wandb is not None:
                        if self.snapshot_histogram:
                            group_key = self._hist_group_key(name).replace("/", "_")
                            entry = hist_accum.setdefault(group_key, {"row": [], "col": []})
                            entry["row"].append(self._maybe_sample(row_energy.detach().float()).cpu())
                            entry["col"].append(self._maybe_sample(col_energy.detach().float()).cpu())
                        if self.snapshot_topk_bar:
                            if row_vals is None or col_vals is None:
                                row_vals, row_idx = _topk_snapshot(row_energy, self.topk)
                                col_vals, col_idx = _topk_snapshot(col_energy, self.topk)
                            row_table = wandb.Table(
                                data=[[str(idx), float(val)] for val, idx in zip(row_vals, row_idx)],
                                columns=["channel", "energy"],
                            )
                            col_table = wandb.Table(
                                data=[[str(idx), float(val)] for val, idx in zip(col_vals, col_idx)],
                                columns=["channel", "energy"],
                            )
                            metrics[f"{self.metric_prefix}/{name}/row_topk_bar"] = wandb.plot.bar(
                                row_table, "channel", "energy"
                            )
                            metrics[f"{self.metric_prefix}/{name}/col_topk_bar"] = wandb.plot.bar(
                                col_table, "channel", "energy"
                            )

        if hist_accum and wandb is not None and self.snapshot_histogram:
            for group_key, values in hist_accum.items():
                row_values = values.get("row", [])
                col_values = values.get("col", [])
                if row_values:
                    merged = torch.cat(row_values).tolist()
                    metrics[f"{self.metric_prefix}/hist/{group_key}/row_energy"] = wandb.Histogram(
                        merged, num_bins=self.snapshot_histogram_bins
                    )
                if col_values:
                    merged = torch.cat(col_values).tolist()
                    metrics[f"{self.metric_prefix}/hist/{group_key}/col_energy"] = wandb.Histogram(
                        merged, num_bins=self.snapshot_histogram_bins
                    )
        if self.snapshot_topk_aggregate:
            for axis in ("row", "col"):
                if topk_accum[axis]:
                    values = torch.cat(topk_accum[axis]).float()
                    metrics[f"{self.metric_prefix}/topk/{axis}_energy_mean"] = values.mean().item()
                    metrics[f"{self.metric_prefix}/topk/{axis}_energy_p95"] = (
                        torch.quantile(values, 0.95).item()
                    )
                    metrics[f"{self.metric_prefix}/topk/{axis}_energy_max"] = values.max().item()
                    if wandb is not None:
                        metrics[f"{self.metric_prefix}/topk/{axis}_energy_hist"] = wandb.Histogram(
                            values.tolist(), num_bins=self.snapshot_histogram_bins
                        )
        return metrics


class QuantWeightL2Probe(MetricProbe):
    def __init__(self, metric_name: str = "monitor/quant/qweight_l2_reg") -> None:
        self.metric_name = metric_name
        self._targets: List[torch.nn.Module] = []

    def bind(self, model: torch.nn.Module) -> None:
        self._targets = [m for m in model.modules() if isinstance(m, IntQuantLinear)]

    def collect(self, model: torch.nn.Module) -> Dict[str, float]:
        cached = getattr(model, "_last_qweight_l2_reg", None)
        if cached is not None:
            return {self.metric_name: float(cached)}
        if not self._targets:
            self.bind(model)
        total_sse = None
        total_numel = 0
        with torch.no_grad():
            for module in self._targets:
                stats = module.weight_l2_stats()
                if stats is None:
                    continue
                sse, numel = stats
                total_sse = sse if total_sse is None else total_sse + sse
                total_numel += numel
        if total_sse is None or total_numel == 0:
            return {}
        total = _to_local_tensor(total_sse / total_numel)
        return {self.metric_name: total.float().item()}


class LossSpikeMeter:
    def __init__(self, ema_decay: float = 0.98, spike_k: float = 3.0) -> None:
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        if spike_k < 0.0:
            raise ValueError(f"spike_k must be >= 0, got {spike_k}")
        self.ema_decay = ema_decay
        self.spike_k = spike_k
        self._ema: Optional[float] = None
        self._ema_sq: Optional[float] = None
        self._num_steps = 0
        self._spike_count = 0

    def state_dict(self) -> Dict[str, float]:
        return {
            "ema_decay": self.ema_decay,
            "spike_k": self.spike_k,
            "ema": self._ema,
            "ema_sq": self._ema_sq,
            "num_steps": self._num_steps,
            "spike_count": self._spike_count,
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.ema_decay = float(state.get("ema_decay", self.ema_decay))
        self.spike_k = float(state.get("spike_k", self.spike_k))
        self._ema = state.get("ema")
        self._ema_sq = state.get("ema_sq")
        self._num_steps = int(state.get("num_steps", 0))
        self._spike_count = int(state.get("spike_count", 0))

    def update(self, loss: float) -> Dict[str, float]:
        loss_val = float(loss)
        if self._ema is None or self._ema_sq is None:
            self._ema = loss_val
            self._ema_sq = loss_val * loss_val
            self._num_steps = 1
            self._spike_count = 0
            return {
                "monitor/loss_ema": self._ema,
                "monitor/loss_std": 0.0,
                "monitor/loss_spike_rate": 0.0,
            }

        var_before = max(self._ema_sq - self._ema * self._ema, 0.0)
        std_before = math.sqrt(var_before)
        threshold = self._ema + self.spike_k * std_before
        if loss_val > threshold:
            self._spike_count += 1
        self._num_steps += 1

        decay = self.ema_decay
        self._ema = self._ema * decay + loss_val * (1.0 - decay)
        self._ema_sq = self._ema_sq * decay + loss_val * loss_val * (1.0 - decay)

        var_after = max(self._ema_sq - self._ema * self._ema, 0.0)
        std_after = math.sqrt(var_after)
        spike_rate = self._spike_count / self._num_steps if self._num_steps > 0 else 0.0
        return {
            "monitor/loss_ema": self._ema,
            "monitor/loss_std": std_after,
            "monitor/loss_spike_rate": spike_rate,
        }


class ForwardConsistencyMeter:
    def __init__(self, history_size: int = 128) -> None:
        if history_size <= 0:
            raise ValueError(f"history_size must be > 0, got {history_size}")
        self.history_size = history_size
        self._delta_history: "deque[float]" = deque(maxlen=history_size)

    def state_dict(self) -> Dict[str, float]:
        return {
            "history_size": self.history_size,
            "delta_history": list(self._delta_history),
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.history_size = int(state.get("history_size", self.history_size))
        values = state.get("delta_history", [])
        self._delta_history = deque((float(v) for v in values), maxlen=self.history_size)

    def update(self, delta_loss: float) -> Dict[str, float]:
        self._delta_history.append(float(delta_loss))
        values = torch.tensor(list(self._delta_history), dtype=torch.float32)
        if values.numel() == 0:
            return {
                "monitor/fc_delta_loss": float(delta_loss),
                "monitor/fc_delta_loss_p95": 0.0,
                "monitor/fc_delta_loss_p99": 0.0,
                "monitor/fc_delta_loss_max": 0.0,
            }
        p95 = torch.quantile(values, 0.95).item()
        p99 = torch.quantile(values, 0.99).item()
        max_val = values.max().item()
        return {
            "monitor/fc_delta_loss": float(delta_loss),
            "monitor/fc_delta_loss_p95": p95,
            "monitor/fc_delta_loss_p99": p99,
            "monitor/fc_delta_loss_max": max_val,
        }


class ProfilingEnvironMeter(EnvironMeter):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        probes: Optional[Sequence[Union[MetricProbe, Callable[[torch.nn.Module], Dict[str, float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._probes = list(probes or [])
        self._extra_metrics: Dict[str, float] = {}
        for probe in self._probes:
            if hasattr(probe, "bind"):
                probe.bind(model)

    def capture_metrics(self) -> None:
        metrics: Dict[str, float] = {}
        for probe in self._probes:
            if hasattr(probe, "collect"):
                metrics.update(probe.collect(self._model))
            else:
                metrics.update(probe(self._model))
        self._extra_metrics = metrics

    def step(self, delta_time: float, global_step: int) -> Dict[str, float]:
        metrics = super().step(delta_time, global_step)
        if self._extra_metrics:
            metrics.update(self._extra_metrics)
            self._extra_metrics = {}
        return metrics
