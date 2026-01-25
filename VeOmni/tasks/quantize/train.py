import json
import math
import os
import time
from datetime import timedelta
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Set

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_energon_dataset,
    build_interleave_dataset,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.ops import loss as veomni_loss
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce

# QAT imports
from EfficientQAT.core.quantizer.config import QuantConfig as EQuantConfig
from EfficientQAT.core.linear.int_quant_linear import (
    IntQuantLinear,
    quantizer_parameters,
    reinit_quant_params,
    sanitize_quant_params,
    set_quant_state
)
from EfficientQAT.core.linear.int_quant_linear_infra import IntQuantLinearInfra, convert_to_infra
from EfficientQAT.core.linear.kernel import int_matmul_backend

from EfficientQAT.core.quantizer.scheduler import (
    RatioBudget,
    TopKSelector,
    QuantizationScheduler,
    GradualQuantController,
    UniformPriorityCalculator,
    MagnitudePriorityCalculator,
    PriorityCalculator,
)

# Support running as package or standalone script
try:
    from .quantizer_arguments import QuantizerArguments
except ImportError:
    # When executed directly (python path/to/train.py), relative imports lack a parent package
    from quantizer_arguments import QuantizerArguments
try:
    from .profiling_utils import (
        ProfilingEnvironMeter,
        GradNormProbe,
        LinearGradStatsProbe,
        LossSpikeMeter,
        ForwardConsistencyMeter,
        QuantWeightL2Probe,
    )
except ImportError:
    from profiling_utils import (
        ProfilingEnvironMeter,
        GradNormProbe,
        LinearGradStatsProbe,
        LossSpikeMeter,
        ForwardConsistencyMeter,
        QuantWeightL2Probe,
    )

logger = helper.create_logger(__name__)


def build_qat_param_groups(model, base_lr, base_weight_decay, quant_lr, quant_weight_decay):
    quant_params = [p for p in quantizer_parameters(model) if p.requires_grad]
    if not quant_params:
        # Check for Infra params: scales, qzeros
        quant_params = []
        for m in model.modules():
            if isinstance(m, IntQuantLinearInfra):
                if hasattr(m, 'scales') and m.scales.requires_grad:
                    quant_params.append(m.scales)
                if hasattr(m, 'qzeros') and m.qzeros.requires_grad:
                    quant_params.append(m.qzeros)
    
    if not quant_params:
        return None
        
    quant_param_ids = {id(p) for p in quant_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in quant_param_ids]
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": base_weight_decay})
    param_groups.append({"params": quant_params, "lr": quant_lr, "weight_decay": quant_weight_decay})
    return param_groups


def _should_skip_module(module_name: str, skip_names: Set[str]) -> bool:
    return any(module_name == skip or module_name.endswith(f".{skip}") for skip in skip_names)


def convert_linear_with_skip(module: torch.nn.Module, prefix: str, config: EQuantConfig, skip_names: Set[str]) -> None:
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if _should_skip_module(child_prefix, skip_names):
            continue
        if isinstance(child, torch.nn.Linear) and not isinstance(child, IntQuantLinear):
            setattr(module, name, IntQuantLinear.from_float(child_prefix, child, config))
        else:
            convert_linear_with_skip(child, child_prefix, config, skip_names)


def freeze_named_modules(model: torch.nn.Module, module_names: Set[str]) -> List[str]:
    frozen = []
    for name, module in model.named_modules():
        if _should_skip_module(name, module_names):
            for param in module.parameters():
                param.requires_grad = False
            frozen.append(name)
    return frozen


def _get_quant_state(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, IntQuantLinear):
            return bool(module.use_weight_quant)
    return False


def _prepare_probe_batch(batch: Dict[str, Any], enable_multisource: bool) -> Dict[str, Any]:
    probe_batch = {k: v for k, v in batch.items()}
    if enable_multisource:
        probe_batch.pop("ds_idx", None)
        probe_batch.pop("source_name", None)
    return {
        k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in probe_batch.items()
    }


def _compute_logit_divergence(
    logits_fp: torch.Tensor,
    logits_q: torch.Tensor,
    labels: torch.Tensor | None,
    metrics: Set[str],
    max_tokens: int,
    temperature: float,
) -> Dict[str, float]:
    if not metrics:
        return {}
    if logits_fp is None or logits_q is None:
        return {}
    if logits_fp.shape != logits_q.shape:
        return {}

    temperature = float(temperature)
    if temperature <= 0.0:
        temperature = 1.0

    logits_fp = logits_fp.float().reshape(-1, logits_fp.shape[-1]) / temperature
    logits_q = logits_q.float().reshape(-1, logits_q.shape[-1]) / temperature
    if labels is not None:
        labels = labels.reshape(-1)
        mask = labels != -100
        if mask.any():
            logits_fp = logits_fp[mask]
            logits_q = logits_q[mask]
        else:
            return {}

    if max_tokens > 0 and logits_fp.shape[0] > max_tokens:
        logits_fp = logits_fp[:max_tokens]
        logits_q = logits_q[:max_tokens]

    log_p_fp = F.log_softmax(logits_fp, dim=-1)
    log_p_q = F.log_softmax(logits_q, dim=-1)
    p_fp = log_p_fp.exp()
    p_q = log_p_q.exp()

    out: Dict[str, float] = {}
    if "kl" in metrics:
        kl = (p_fp * (log_p_fp - log_p_q)).sum(dim=-1).mean().item()
        out["monitor/fc_logit_kl"] = kl * (temperature ** 2)
    if "js" in metrics:
        m = 0.5 * (p_fp + p_q)
        log_m = torch.log(m + 1e-8)
        js = 0.5 * (
            (p_fp * (log_p_fp - log_m)).sum(dim=-1) + (p_q * (log_p_q - log_m)).sum(dim=-1)
        )
        out["monitor/fc_logit_js"] = js.mean().item() * (temperature ** 2)
    return out


def _compute_weight_l2_reg_loss(model: torch.nn.Module) -> torch.Tensor | None:
    total_sse = None
    total_numel = 0
    for module in model.modules():
        if isinstance(module, IntQuantLinear):
            stats = module.weight_l2_stats()
            if stats is None:
                continue
            sse, numel = stats
            total_sse = sse if total_sse is None else total_sse + sse
            total_numel += numel
    if total_sse is None or total_numel == 0:
        return None
    reg_loss = total_sse / total_numel
    return _to_local_tensor(reg_loss)


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
        if hasattr(tensor, "wait"):
            tensor = tensor.wait()
    return tensor


def _compute_kd_loss(
    student_logits: torch.Tensor | None,
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor | None:
    if student_logits is None or teacher_logits is None:
        return None

    temperature = float(temperature)
    if temperature <= 0.0:
        temperature = 1.0

    student_logits = student_logits.float().reshape(-1, student_logits.shape[-1]) / temperature
    teacher_logits = teacher_logits.float().reshape(-1, teacher_logits.shape[-1]) / temperature
    if student_logits.shape != teacher_logits.shape:
        return None
    if labels is not None:
        labels = labels.reshape(-1)
        mask = labels != -100
        if mask.any():
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
        else:
            return None

    kd_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )
    return _to_local_tensor(kd_loss * (temperature ** 2))


def _get_kd_skip_reason(
    student_logits: torch.Tensor | None,
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor | None,
) -> str | None:
    if student_logits is None:
        return "student logits missing"
    if teacher_logits is None:
        return "teacher logits missing"
    if student_logits.shape != teacher_logits.shape:
        return (
            "logits shape mismatch "
            f"(student={tuple(student_logits.shape)}, teacher={tuple(teacher_logits.shape)})"
        )
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.reshape(-1)
        mask = labels != -100
        if not mask.any().item():
            return "labels all -100 after mask"
    return None


def _maybe_get_output_keys(output: object) -> str | None:
    if not hasattr(output, "keys"):
        return None
    try:
        keys = list(output.keys())
    except Exception:
        return None
    if not keys:
        return None
    preview = ", ".join(str(k) for k in keys[:20])
    if len(keys) > 20:
        preview = f"{preview}, ..."
    return f"[{preview}]"


def _matches_name(name: str, patterns: List[str]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)


def _resolve_quant_modules(
    model: torch.nn.Module, module_names: List[str]
) -> List[tuple[str, IntQuantLinear]]:
    named = dict(model.named_modules())
    resolved: List[tuple[str, IntQuantLinear]] = []
    if not module_names:
        return resolved
    if module_names:
        for name in module_names:
            module = named.get(name)
            if isinstance(module, IntQuantLinear):
                resolved.append((name, module))
        if resolved:
            return resolved
        for name, module in named.items():
            if isinstance(module, IntQuantLinear) and _matches_name(name, module_names):
                resolved.append((name, module))
        return resolved
    return resolved


def _percentile_label(p: float) -> str:
    return f"p{int(round(p * 1000.0))}"


def _compute_quant_stats_for_module(
    module: IntQuantLinear,
    percentiles: List[float],
    prev_scale: torch.Tensor | None,
    prev_weight: torch.Tensor | None,
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor] | None:
    q = getattr(module, "weight_quantizer", None)
    if q is None or not hasattr(q, "scale") or not hasattr(q, "zero_point"):
        return None
    group_size = getattr(q, "group_size", None)
    if group_size is None or int(group_size) <= 0:
        return None

    with torch.no_grad():
        weight = _to_local_tensor(module.weight.detach())
        scale = _to_local_tensor(q.scale.detach())
        zero_point = _to_local_tensor(q.zero_point.detach())
        scale, round_zero_point = q.cal_qparams(scale, zero_point)
        x = weight.reshape(-1, int(group_size))
        scale = scale.reshape(-1, 1)
        round_zero_point = round_zero_point.reshape(-1, 1)

        mask = None
        if hasattr(q, "group_mask") and q.group_mask is not None:
            mask = q.group_mask
            if mask.numel() != x.shape[0]:
                mask = None
            else:
                mask = mask.to(device=x.device, dtype=torch.bool)
        elif hasattr(q, "_split_quant_groups"):
            qg = int(q._split_quant_groups(x))
            qg = max(min(qg, x.shape[0]), 0)
            mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
            if qg > 0:
                mask[:qg] = True
        if mask is not None:
            if not mask.any():
                return None
            x = x[mask]
            scale = scale[mask]
            round_zero_point = round_zero_point[mask]

        x_int_raw = torch.round(x / scale) + round_zero_point
        clip_mask = (x_int_raw < q.qmin) | (x_int_raw > q.qmax)
        x_int = torch.clamp(x_int_raw, q.qmin, q.qmax)
        sat_mask = (x_int == q.qmin) | (x_int == q.qmax)

        abs_x = x.abs()
        abs_max = abs_x.max().item()
        clip_rate = clip_mask.float().mean().item()
        sat_rate = sat_mask.float().mean().item()

        stats: Dict[str, float] = {
            "clip_rate": clip_rate,
            "saturation_rate": sat_rate,
            "weight_abs_max": abs_max,
        }
        for p in percentiles:
            if 0.0 < p < 1.0:
                stats[f"weight_abs_{_percentile_label(p)}"] = torch.quantile(abs_x, p).item()

        weight_flat = x.reshape(-1).float()
        quantized = (x_int - round_zero_point) * scale
        quant_residual = (quantized - x).reshape(-1).float()
        quant_norm = torch.norm(quant_residual).item()
        update_norm = 0.0
        if prev_weight is not None and prev_weight.shape == weight_flat.shape:
            prev_weight = prev_weight.to(device=weight_flat.device, dtype=weight_flat.dtype)
            update_norm = torch.norm(weight_flat - prev_weight).item()
        stats["qur"] = quant_norm / (update_norm + 1e-6)

        scale_flat = scale.reshape(-1).float().cpu()
        scale_delta = 0.0
        if prev_scale is not None and prev_scale.shape == scale_flat.shape:
            scale_delta = (scale_flat - prev_scale).abs().mean().item()
        stats["scale_abs_delta_mean"] = scale_delta
        weight_snapshot = weight_flat.detach().float().cpu()
        return stats, scale_flat, weight_snapshot


def _compute_quant_stat_metrics(
    modules: List[tuple[str, IntQuantLinear]],
    percentiles: List[float],
    prev_scales: Dict[str, torch.Tensor],
    prev_weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, module in modules:
        stats = _compute_quant_stats_for_module(
            module, percentiles, prev_scales.get(name), prev_weights.get(name)
        )
        if stats is None:
            continue
        stat_dict, scale_snapshot, weight_snapshot = stats
        prev_scales[name] = scale_snapshot
        prev_weights[name] = weight_snapshot
        for key, value in stat_dict.items():
            metrics[f"monitor/quant/{name}/{key}"] = value
    return metrics


def _forward_consistency_probe(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    model_fwd_context,
    logit_metrics: Set[str],
    logit_max_tokens: int,
    logit_temperature: float,
) -> Dict[str, float]:
    was_training = model.training
    quant_enabled = _get_quant_state(model)
    try:
        model.train(False)
        with torch.no_grad():
            set_quant_state(model, weight_quant=False)
            with model_fwd_context:
                out_fp = model(**batch, use_cache=False)
            loss_fp = out_fp.loss.mean().item()
            logits_fp = getattr(out_fp, "logits", None)

            set_quant_state(model, weight_quant=True)
            with model_fwd_context:
                out_q = model(**batch, use_cache=False)
            loss_q = out_q.loss.mean().item()
            logits_q = getattr(out_q, "logits", None)
    except Exception as exc:
        logger.warning_rank0("Forward consistency probe failed: %s", exc)
        return {}
    finally:
        set_quant_state(model, weight_quant=quant_enabled)
        model.train(was_training)

    metrics = {"_delta_loss": loss_q - loss_fp}
    labels = batch.get("labels") if isinstance(batch, dict) else None
    if labels is not None and not isinstance(labels, torch.Tensor):
        labels = None
    metrics.update(
        _compute_logit_divergence(
            logits_fp,
            logits_q,
            labels,
            logit_metrics,
            logit_max_tokens,
            logit_temperature,
        )
    )
    return metrics


def _sync_gradual_quantizer_metadata(model: torch.nn.Module) -> int:
    """
    When the model is initialized on `meta` (common for FSDP), quantizers may cache device/size metadata
    before real weights are materialized. This sync keeps gradual scheduling working after weights load.
    """
    updated = 0
    for m in model.modules():
        if not isinstance(m, IntQuantLinear) or getattr(m, "weight_quantizer", None) is None:
            continue
        q = m.weight_quantizer
        if hasattr(q, "_device"):
            q._device = m.weight.device
            updated += 1
        if hasattr(q, "_num_elements"):
            # Prefer saved full size when available (e.g., before FSDP sharding).
            if hasattr(q, "_num_elements_full"):
                q._num_elements = q._num_elements_full
            else:
                # For DTensor, numel() can be local; prefer global shape product if available.
                try:
                    shape_numel = 1
                    for dim in m.weight.shape:
                        shape_numel *= int(dim)
                except Exception:
                    shape_numel = m.weight.numel()
                if hasattr(m.weight, "to_local") and shape_numel != m.weight.numel():
                    q._num_elements = shape_numel
                else:
                    q._num_elements = m.weight.numel()
            updated += 1
    return updated


@dataclass
class ProfileArguments:
    profile_module_name: List[str] = field(
        default_factory=lambda: ["model.layers.0", "layers.0", "layer.0"],
        metadata={"help": "Module names to profile (e.g., ['model.layers.0'])."},
    )
    loss_spike_k: float = field(
        default=3.0,
        metadata={"help": "Spike threshold k for loss_t > EMA(loss) + k*std."},
    )
    loss_spike_ema_decay: float = field(
        default=0.98,
        metadata={"help": "EMA decay for loss spike rate calculation."},
    )
    forward_consistency_interval: int = field(
        default=0,
        metadata={"help": "Steps between forward consistency probes. 0 disables."},
    )
    forward_consistency_history_size: int = field(
        default=128,
        metadata={"help": "History size for delta loss tail stats."},
    )
    forward_consistency_logit_metrics: List[str] = field(
        default_factory=lambda: ["kl", "js"],
        metadata={"help": "Logit divergence metrics to compute, e.g. ['kl','js']."},
    )
    forward_consistency_logit_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for logit divergence metrics (T)."},
    )
    forward_consistency_logit_max_tokens: int = field(
        default=128,
        metadata={"help": "Max tokens to use when computing logit divergence. 0 for full."},
    )
    quant_stat_interval: int = field(
        default=0,
        metadata={"help": "Steps between quant stats collection. 0 disables."},
    )
    quant_stat_module_name: List[str] = field(
        default_factory=list,
        metadata={"help": "Module names to collect quant stats; empty disables."},
    )
    quant_stat_percentiles: List[float] = field(
        default_factory=lambda: [0.999],
        metadata={"help": "Percentiles to report for abs(weight), e.g. [0.999]."},
    )
    linear_grad_interval: int = field(
        default=0,
        metadata={"help": "Steps between linear grad stats collection. 0 disables."},
    )
    linear_grad_snapshot_interval: int = field(
        default=0,
        metadata={"help": "Steps between row/col energy top-k snapshots. 0 disables."},
    )
    linear_grad_snapshot_steps: List[int] = field(
        default_factory=list,
        metadata={"help": "Explicit steps for row/col energy top-k snapshots."},
    )
    linear_grad_snapshot_histogram: bool = field(
        default=False,
        metadata={"help": "Log row/col energy histograms on snapshot steps."},
    )
    linear_grad_snapshot_histogram_bins: int = field(
        default=64,
        metadata={"help": "Number of bins for row/col energy histograms."},
    )
    linear_grad_snapshot_topk_bar: bool = field(
        default=False,
        metadata={"help": "Log row/col top-k bar plots on snapshot steps."},
    )
    linear_grad_snapshot_histogram_group_by: str = field(
        default="module",
        metadata={"help": "Histogram aggregation: module | layer | suffix."},
    )
    linear_grad_snapshot_histogram_sample: int = field(
        default=0,
        metadata={"help": "Sample size per module for histogram aggregation. 0 keeps all."},
    )
    linear_grad_snapshot_topk_aggregate: bool = field(
        default=False,
        metadata={"help": "Aggregate top-k values across modules on snapshot steps."},
    )
    linear_grad_topk: int = field(
        default=20,
        metadata={"help": "Top-k channels for row/col energy ratios and snapshots."},
    )
    linear_grad_spectral_iters: int = field(
        default=2,
        metadata={"help": "Power iterations for spectral norm approximation. 0 disables."},
    )
    linear_grad_sign_flip_sample: int = field(
        default=65536,
        metadata={"help": "Sample size for sign flip rate; 0 uses full tensor."},
    )


@dataclass
class DistillArguments:
    kd_mode: Literal["none", "logits"] = field(
        default="none",
        metadata={"help": "Knowledge distillation mode: none | logits."},
    )
    teacher_model: str | None = field(
        default=None,
        metadata={"help": "Path to teacher model weights/config for KD."},
    )
    kd_alpha: float = field(
        default=0.5,
        metadata={"help": "KD loss weight (alpha)."},
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={"help": "KD temperature for logits."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    quantizer: "QuantizerArguments" = field(default_factory=QuantizerArguments)
    profile: "ProfileArguments" = field(default_factory=ProfileArguments)
    distill: "DistillArguments" = field(default_factory=DistillArguments)

def main():
    dist.init_process_group(backend=get_nccl_backend(), timeout=timedelta(minutes=30))
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.enable_multisource:
            logger.info_rank0("Start building interleave dataset")
            train_dataset = build_interleave_dataset(
                args.data.train_path, args.data.datasets_type, transform=transform, seed=args.train.seed
            )
        elif args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "energon":
            logger.info_rank0("Start building Megatron-Energon native dataset")
            train_dataset = build_energon_dataset(
                args.data.train_path,
                transform=transform,
                max_samples_per_sequence=args.data.max_samples_per_sequence
                if hasattr(args.data, "max_samples_per_sequence")
                else None,
                virtual_epoch_length=args.data.virtual_epoch_length
                if hasattr(args.data, "virtual_epoch_length")
                else None,
                shuffle_buffer_size=args.data.shuffle_buffer_size
                if hasattr(args.data, "shuffle_buffer_size")
                else None,
                num_workers=args.data.num_workers,
            )
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    # 将 CLI/YAML 量化参数映射到 EfficientQAT 的 QuantConfig，并替换线性层
    qcfg = EQuantConfig(
        quant_type=args.quantizer.quant_type,
        n_bits=args.quantizer.n_bits,
        group_size=args.quantizer.group_size,
        clamp_method=args.quantizer.clamp_method,
        round_method=args.quantizer.round_method,
        stat_quant=args.quantizer.stat_quant,
        iterative_freezing=args.quantizer.iterative_freezing,
        iterative_freezing_sheduler=args.quantizer.iterative_freezing_sheduler,
        is_tracking=args.quantizer.is_tracking,
        freeze_momentum=args.quantizer.freeze_momentum,
        freeze_threshold=args.quantizer.freeze_threshold,
        interpolate=args.quantizer.interpolate,
        lora_rank=args.quantizer.lora_rank,
        decay_rate=args.quantizer.decay_rate,
        shrinking_ratio=args.quantizer.shrinking_ratio,
        ramp_len=args.quantizer.ramp_len,
        ramp_mode=args.quantizer.ramp_mode,
        ramp_sigmoid_a=args.quantizer.ramp_sigmoid_a,
    )
    skip_quant_modules = {"lm_head","embed_tokens"}
    convert_linear_with_skip(model, prefix="", config=qcfg, skip_names=skip_quant_modules)
    
    # 转换到 Infra 模式 (如果启用)
    if getattr(args.quantizer, "enable_infra", False):
        logger.info_rank0("Converting model to IntQuantLinearInfra mode.")
        convert_to_infra(model, kernel_backend=int_matmul_backend)

    frozen_modules = freeze_named_modules(model, skip_quant_modules)
    if frozen_modules:
        logger.info_rank0("Skip quantization and freeze modules: %s", ", ".join(sorted(frozen_modules)))
    else:
        logger.info_rank0("Skip list did not match any module: %s", ", ".join(sorted(skip_quant_modules)))
    logger.info_rank0(
        "QAT layer conversion done: IntQuantLinear=%d, remaining nn.Linear=%d",
        sum(1 for m in model.modules() if isinstance(m, IntQuantLinear)),
        sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear) and not isinstance(m, IntQuantLinear)),
    )
# <------QAT模型构建设计-------->
# 计划：在构建完成后读取QAT配置（wbits/group_size/quant_type），调用 EfficientQAT 的 convert_linear 将 nn.Linear 替换为 IntQuantLinear/QuantLinearFake，并挂载 weight_quantizer。
# 预留：可选逐层白名单/正则匹配，或从 YAML/CLI 指定需要量化的模块列表，默认跳过嵌入/头部。
# 设备：保持 init_device=meta/cuda 的流程不变，转换后权重依然沿用原参数设备和 dtype。
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    reinit_quant_params(model)
    sanitize_quant_params(model)
    
    # Gradual sync only for non-infra (since infra has no quantizer object)
    if not getattr(args.quantizer, "enable_infra", False):
        _sync_gradual_quantizer_metadata(model)
        logger.info_rank0("Reinitialized quantizer params from loaded weights.")
    else:
        logger.info_rank0("Skipping gradual metadata sync for Infra mode.")

    kd_mode = args.distill.kd_mode
    kd_enabled = kd_mode != "none"
    if kd_enabled and veomni_loss.fused_linear_cross_entropy is not None:
        # KD needs student logits; disable fused loss kernels which skip logits.
        logger.info_rank0("KD enabled: disabling fused_linear_cross_entropy to keep logits.")
        veomni_loss.fused_linear_cross_entropy = None
    kd_alpha = float(args.distill.kd_alpha)
    kd_temperature = float(args.distill.kd_temperature)
    if kd_alpha < 0.0 or kd_alpha > 1.0:
        logger.warning_rank0("distill.kd_alpha=%s is out of [0,1]; clamping.", kd_alpha)
        kd_alpha = max(0.0, min(1.0, kd_alpha))
    if kd_temperature <= 0.0:
        logger.warning_rank0("distill.kd_temperature=%s is <= 0; defaulting to 1.0.", kd_temperature)
        kd_temperature = 1.0

    teacher_model = None
    if kd_enabled:
        if not args.distill.teacher_model:
            raise ValueError("distill.teacher_model must be set when distill.kd_mode is not 'none'.")
        logger.info_rank0(
            "Prepare teacher model for KD: mode=%s, alpha=%s, temperature=%s, path=%s",
            kd_mode,
            kd_alpha,
            kd_temperature,
            args.distill.teacher_model,
        )
        teacher_model = build_foundation_model(
            config_path=args.distill.teacher_model,
            weights_path=args.distill.teacher_model,
            torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            init_device=args.train.init_device,
            force_use_huggingface=args.model.force_use_huggingface,
        )
        teacher_model = build_parallelize_model(
            teacher_model,
            init_device=args.train.init_device,
            weights_path=args.distill.teacher_model,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_gradient_checkpointing=False,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            basic_modules=teacher_model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
        )
        teacher_model.eval()
        teacher_model.requires_grad_(False)
    # <------QAT并行与开关管理设计-------->
    # 设置量化开关：初始化时 use_weight_quant=False，用于前期预热；在 warmup_steps 后切换为 True。
    # 渐进量化：如果使用 GradualQuantizer，创建 GradualQuantContext(total_steps=train_steps*num_epochs, warmup_steps=qat_warmup)，在训练循环中 step(step_id) 更新 quantization_position_ratio。
    # 激活假量化：若引入 activation observer，可在 model_fwd_context 前后包一层 autocast + fake_quant_hook。

    quant_lr = args.quantizer.quant_lr if args.quantizer.quant_lr is not None else args.train.lr
    quant_weight_decay = args.quantizer.quant_weight_decay
    qat_param_groups = build_qat_param_groups(
        model,
        base_lr=args.train.lr,
        base_weight_decay=args.train.weight_decay,
        quant_lr=quant_lr,
        quant_weight_decay=quant_weight_decay,
    )
    if qat_param_groups is not None:
        logger.info_rank0(
            "QAT param groups: %d base params, %d quant params (quant_lr=%s, quant_wd=%s)",
            len(qat_param_groups[0]["params"]) if len(qat_param_groups) > 1 else 0,
            len(qat_param_groups[-1]["params"]),
            quant_lr,
            quant_weight_decay,
        )

    qweight_l2_reg_lambda = float(args.quantizer.qweight_l2_reg_lambda)
    if qweight_l2_reg_lambda < 0.0:
        logger.warning_rank0(
            "qweight_l2_reg_lambda=%s is negative; clamping to 0.",
            qweight_l2_reg_lambda,
        )
        qweight_l2_reg_lambda = 0.0
    enable_qweight_l2_reg = (
        args.quantizer.enable_qweight_l2_reg
        and qat_param_groups is not None
        and qweight_l2_reg_lambda > 0.0
    )
    if args.quantizer.enable_qweight_l2_reg and qat_param_groups is None:
        logger.info_rank0("QWeight L2 reg requested but no quantizer params found; disabled.")
    elif args.quantizer.enable_qweight_l2_reg and qweight_l2_reg_lambda <= 0.0:
        logger.info_rank0("QWeight L2 reg requested but qweight_l2_reg_lambda <= 0; disabled.")
    if enable_qweight_l2_reg:
        logger.info_rank0("QWeight L2 reg enabled with lambda=%s", qweight_l2_reg_lambda)

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        param_groups=qat_param_groups,
    )
# <------QAT参数分组设计-------->
# 计划拆分参数组：1) 主权重（低 lr/或与原配置一致）；
# 2) 量化参数 scale/zero_point（单独 lr、weight_decay=0）；必要时冻结部分权重仅训练量化参数。
# 可通过 EfficientQAT.core.linear.int_quant_linear.{weight_parameters,quant_parameters} 辅助函数构建 param_groups。
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={
                    **vars(args.model),
                    **vars(args.data),
                    **vars(args.train),
                    **vars(args.quantizer),
                    **vars(args.distill),
                },  # flatten dict
            )

        # save model_assets before training
        model_assets = [model_config, tokenizer if args.data.data_type == "plaintext" else chat_template]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    grad_probes = []
    if args.profile.profile_module_name:
        grad_probes.append(GradNormProbe(module_names=args.profile.profile_module_name, metric_prefix="monitor/grad_norm"))
    linear_grad_probe = None
    if args.profile.profile_module_name and args.profile.linear_grad_interval > 0:
        linear_grad_probe = LinearGradStatsProbe(
            module_names=args.profile.profile_module_name,
            metric_prefix="monitor/linear_grad",
            interval=args.profile.linear_grad_interval,
            snapshot_interval=args.profile.linear_grad_snapshot_interval,
            snapshot_steps=args.profile.linear_grad_snapshot_steps,
            snapshot_histogram=args.profile.linear_grad_snapshot_histogram,
            snapshot_histogram_bins=args.profile.linear_grad_snapshot_histogram_bins,
            snapshot_topk_bar=args.profile.linear_grad_snapshot_topk_bar,
            snapshot_histogram_group_by=args.profile.linear_grad_snapshot_histogram_group_by,
            snapshot_histogram_sample=args.profile.linear_grad_snapshot_histogram_sample,
            snapshot_topk_aggregate=args.profile.linear_grad_snapshot_topk_aggregate,
            topk=args.profile.linear_grad_topk,
            spectral_iters=args.profile.linear_grad_spectral_iters,
            sign_flip_sample=args.profile.linear_grad_sign_flip_sample,
        )
        grad_probes.append(linear_grad_probe)
    if enable_qweight_l2_reg:
        grad_probes.append(QuantWeightL2Probe(metric_name="monitor/quant/qweight_l2_reg"))
    environ_meter = ProfilingEnvironMeter(
        model=model,
        probes=grad_probes,
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )
    loss_spike_meter = LossSpikeMeter(
        ema_decay=args.profile.loss_spike_ema_decay,
        spike_k=args.profile.loss_spike_k,
    )
    forward_consistency_meter = ForwardConsistencyMeter(
        history_size=args.profile.forward_consistency_history_size
    )
    logit_metric_allowlist = {"kl", "js"}
    forward_logit_metrics = {m.lower() for m in args.profile.forward_consistency_logit_metrics}
    forward_logit_metrics = {m for m in forward_logit_metrics if m in logit_metric_allowlist}
    if args.profile.forward_consistency_logit_metrics and not forward_logit_metrics:
        logger.warning_rank0(
            "forward_consistency_logit_metrics has no valid entries. Allowed: %s",
            ", ".join(sorted(logit_metric_allowlist)),
        )
    quant_stat_interval = int(args.profile.quant_stat_interval)
    quant_stat_module_names = [name for name in args.profile.quant_stat_module_name if name]
    quant_stat_percentiles = [float(p) for p in args.profile.quant_stat_percentiles if 0.0 < p < 1.0]
    if args.profile.quant_stat_percentiles and not quant_stat_percentiles:
        logger.warning_rank0("quant_stat_percentiles has no valid entries in (0,1).")
        quant_stat_percentiles = [0.999]
    enable_quant_stats = quant_stat_interval > 0 and bool(quant_stat_module_names)
    if quant_stat_interval > 0 and not quant_stat_module_names:
        logger.info_rank0("Quant stats disabled because quant_stat_module_name is empty.")
    quant_stat_modules: List[tuple[str, IntQuantLinear]] | None = None
    quant_stat_prev_scales: Dict[str, torch.Tensor] = {}
    quant_stat_prev_weights: Dict[str, torch.Tensor] = {}
    kd_warning_emitted = False

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        if "loss_spike_meter" in state["extra_state"]:
            loss_spike_meter.load_state_dict(state["extra_state"]["loss_spike_meter"])
        if "forward_consistency_meter" in state["extra_state"]:
            forward_consistency_meter.load_state_dict(state["extra_state"]["forward_consistency_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
        )
    model.train()

    # <------QAT Gradual Quantization Integration-------->
    # Initialize gradual quantization controller if enabled
    # gradual 量化：全程 use_weight_quant=True，通过 group_mask 控制哪些 groups 被量化
    gradual_controller = None
    enable_infra = getattr(args.quantizer, "enable_infra", False)
    
    if args.quantizer.enable_gradual_quant and args.quantizer.quant_type == "gradual" and not enable_infra:
        logger.info_rank0("Initializing GradualQuantController with gradual quantization")
        total_steps = args.train.train_steps * args.train.num_train_epochs
        warmup_steps = args.quantizer.qat_warmup_steps
        start_ratio = args.quantizer.gradual_start_ratio
        end_ratio = args.quantizer.gradual_end_ratio
        # Interpret gradual_end_ratio as the fraction of training steps to reach full quantization.
        end_step_ratio = float(end_ratio)
        if end_step_ratio <= 0.0:
            logger.warning_rank0(
                "gradual_end_ratio=%s <= 0; defaulting to 1.0 (full quantization by end of training).",
                end_ratio,
            )
            end_step_ratio = 1.0
        end_step_ratio = min(end_step_ratio, 1.0)
        ramp_end_step = max(int(total_steps * end_step_ratio), warmup_steps + 1)

        # 选择优先级计算器
        priority_type = args.quantizer.priority_type
        if priority_type == "uniform":
            priority_calculator = UniformPriorityCalculator()
        elif priority_type == "magnitude":
            priority_calculator = MagnitudePriorityCalculator()
        else:
            logger.info_rank0(f"Unknown priority_type={priority_type}, fallback to uniform")
            priority_calculator = UniformPriorityCalculator()

        budget = RatioBudget(
            start_ratio=start_ratio,
            end_ratio=1.0,
            total_steps=ramp_end_step,
            warmup_steps=warmup_steps,
        )
        selector = TopKSelector()
        scheduler = QuantizationScheduler(
            budget_policy=budget,
            selector=selector,
            priority_calculator=priority_calculator,
        )
        gradual_controller = GradualQuantController(model, scheduler)

        # Gradual quantization: quantizer is always enabled, but group_mask controls which groups are quantized
        set_quant_state(model, weight_quant=True)
        logger.info_rank0(
            f"Gradual quantization enabled: warmup_steps={warmup_steps}, "
            f"start_ratio={start_ratio}, end_ratio={end_ratio} (ramp_end_step={ramp_end_step}/{total_steps}), "
            f"priority_type={priority_type}, quantizer always active"
        )
    else:
        # Non-gradual mode: enable quantization from the start
        set_quant_state(model, weight_quant=True)
        logger.info_rank0("Quantization enabled from the start")

    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    # <------QAT训练循环设计-------->
    # 在 epoch/step 循环外部：with GradualQuantContext(model, total_steps=args.train.train_steps*args.train.num_train_epochs, warmup_steps=qat_warmup) as qat_sched:
    # 在每个 global_step 开始时调用 qat_sched.step(global_step) 以更新 ratio；在 warmup 结束时 set_quant_state(model, weight_quant=True) 启用假量化。
    # 可选：按 step 动态调整 n_bits/group_size（如后续扩展）或只更新 quantization_position_ratio。
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            total_task_loss = 0.0
            total_kd_loss = 0.0
            total_qweight_l2_reg = 0.0
            total_qweight_l2_scaled = 0.0
            debug_batch = None
            synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                if debug_batch is None:
                    debug_batch = {k: v for k, v in micro_batch.items()}
                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    student_out = model(**micro_batch, use_cache=False)
                    task_loss = student_out.loss.mean()
                    task_loss = _to_local_tensor(task_loss)
                    combined_loss = task_loss
                    kd_loss = None
                    if teacher_model is not None:
                        teacher_inputs = {k: v for k, v in micro_batch.items() if k != "labels"}
                        with torch.no_grad():
                            teacher_out = teacher_model(**teacher_inputs, use_cache=False)
                        kd_loss = _compute_kd_loss(
                            student_logits=getattr(student_out, "logits", None),
                            teacher_logits=getattr(teacher_out, "logits", None),
                            labels=micro_batch.get("labels"),
                            temperature=kd_temperature,
                        )
                        if kd_loss is not None:
                            combined_loss = (1.0 - kd_alpha) * task_loss + kd_alpha * kd_loss
                        elif not kd_warning_emitted and args.train.global_rank == 0:
                            kd_skip_reason = _get_kd_skip_reason(
                                getattr(student_out, "logits", None),
                                getattr(teacher_out, "logits", None),
                                micro_batch.get("labels"),
                            )
                            student_keys = _maybe_get_output_keys(student_out)
                            teacher_keys = _maybe_get_output_keys(teacher_out)
                            kd_skip_msg = (
                                f"KD enabled but {kd_skip_reason}; skipping KD loss."
                                if kd_skip_reason
                                else "KD enabled but logits missing or mismatched; skipping KD loss."
                            )
                            if student_keys:
                                kd_skip_msg = f"{kd_skip_msg} student_out.keys={student_keys}"
                            if teacher_keys:
                                kd_skip_msg = f"{kd_skip_msg} teacher_out.keys={teacher_keys}"
                            logger.warning_rank0(
                                kd_skip_msg
                            )
                            kd_warning_emitted = True
                    if enable_qweight_l2_reg:
                        reg_loss = _compute_weight_l2_reg_loss(model)
                        if reg_loss is not None:
                            combined_loss = combined_loss + qweight_l2_reg_lambda * reg_loss
                            reg_loss_val = float(reg_loss.detach())
                            total_qweight_l2_reg += reg_loss_val / len(micro_batches)
                            total_qweight_l2_scaled += (
                                qweight_l2_reg_lambda * reg_loss_val / len(micro_batches)
                            )
                    loss: "torch.Tensor" = combined_loss / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                total_task_loss += task_loss.item() / len(micro_batches)
                if kd_loss is not None:
                    total_kd_loss += kd_loss.item() / len(micro_batches)
                del micro_batch

            # Prefer model-provided clip_grad_norm_ (now both FSDP1 and FSDP2 registers custom grad norm clipping)
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                logger.info_rank0(
                    "Can NOT find regitsered clip_grad_norm_ method in the model, using PyTorch default implementation.."
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            local_non_finite = 0
            if not math.isfinite(total_loss) or not math.isfinite(grad_norm):
                local_non_finite = 1
            non_finite = all_reduce(local_non_finite, op="max", group=get_parallel_state().fsdp_group)
            if non_finite > 0:
                debug_dir = os.path.join(args.train.output_dir, "nan_debug", f"global_step_{global_step}")
                if local_non_finite > 0 and debug_batch is not None:
                    os.makedirs(debug_dir, exist_ok=True)
                    batch_path = os.path.join(debug_dir, f"micro_batch_rank{args.train.global_rank}.pt")
                    torch.save(
                        {
                            "micro_batch": debug_batch,
                            "global_step": global_step,
                            "rank": args.train.global_rank,
                        },
                        batch_path,
                    )
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "loss_spike_meter": loss_spike_meter.state_dict(),
                        "forward_consistency_meter": forward_consistency_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(os.path.join(debug_dir, "checkpoints"), state, global_steps=global_step)
                logger.warning_rank0(
                    "Detected non-finite loss/grad_norm. Saved debug state to %s; stopping training.",
                    debug_dir,
                )
                raise RuntimeError("Non-finite loss/grad_norm detected.")

            if enable_qweight_l2_reg:
                model._last_qweight_l2_reg = total_qweight_l2_reg
            if linear_grad_probe is not None:
                linear_grad_probe.set_step(global_step)
            environ_meter.capture_metrics()
            forward_metrics: Dict[str, float] = {}
            if (
                args.profile.forward_consistency_interval > 0
                and debug_batch is not None
                and global_step % args.profile.forward_consistency_interval == 0
            ):
                probe_batch = _prepare_probe_batch(debug_batch, args.data.enable_multisource)
                probe_metrics = _forward_consistency_probe(
                    model=model,
                    batch=probe_batch,
                    model_fwd_context=model_fwd_context,
                    logit_metrics=forward_logit_metrics,
                    logit_max_tokens=args.profile.forward_consistency_logit_max_tokens,
                    logit_temperature=args.profile.forward_consistency_logit_temperature,
                )
                if probe_metrics:
                    delta_loss = probe_metrics.pop("_delta_loss", None)
                    if delta_loss is not None:
                        meter_metrics = forward_consistency_meter.update(delta_loss)
                        forward_metrics.update(meter_metrics)
                        forward_metrics["monitor/fc_delta_nll"] = float(meter_metrics["monitor/fc_delta_loss"])
                        forward_metrics["monitor/fc_delta_nll_p95"] = float(
                            meter_metrics["monitor/fc_delta_loss_p95"]
                        )
                        forward_metrics["monitor/fc_delta_nll_p99"] = float(
                            meter_metrics["monitor/fc_delta_loss_p99"]
                        )
                        forward_metrics["monitor/fc_delta_nll_max"] = float(
                            meter_metrics["monitor/fc_delta_loss_max"]
                        )
                    forward_metrics.update(probe_metrics)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            quant_stat_metrics: Dict[str, float] = {}
            if (
                enable_quant_stats
                and args.train.global_rank == 0
                and quant_stat_interval > 0
                and global_step % quant_stat_interval == 0
            ):
                if quant_stat_modules is None:
                    quant_stat_modules = _resolve_quant_modules(model, quant_stat_module_names)
                    if not quant_stat_modules:
                        logger.info_rank0("Quant stats enabled but no IntQuantLinear modules matched.")
                if quant_stat_modules:
                    quant_stat_metrics = _compute_quant_stat_metrics(
                        quant_stat_modules,
                        quant_stat_percentiles,
                        quant_stat_prev_scales,
                        quant_stat_prev_weights,
                    )

            # <------QAT Gradual Quantization Step Update-------->
            # Update gradual quantization state: group_mask controls which groups are quantized
            # 在 gradual 模式下，全程 use_weight_quant=True，但通过 group_mask 控制哪些 groups 被量化
            if gradual_controller is not None:
                gradual_controller.on_step_end(step=global_step, epoch=epoch)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm, total_task_loss, total_kd_loss = all_reduce(
                (total_loss, grad_norm, total_task_loss, total_kd_loss),
                group=get_parallel_state().fsdp_group,
            )
            qweight_metrics: Dict[str, float] = {}
            if enable_qweight_l2_reg:
                qweight_l2_reg, qweight_l2_scaled = all_reduce(
                    (total_qweight_l2_reg, total_qweight_l2_scaled),
                    group=get_parallel_state().fsdp_group,
                )
                qweight_metrics = {
                    "monitor/quant/qweight_l2_reg": qweight_l2_reg,
                    "monitor/quant/qweight_l2_lambda": qweight_l2_reg_lambda,
                    "monitor/quant/qweight_l2_scaled": qweight_l2_scaled,
                }
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            spike_metrics = loss_spike_meter.update(total_loss)
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
            train_metrics.update(spike_metrics)
            if qweight_metrics:
                train_metrics.update(qweight_metrics)
            if quant_stat_metrics:
                train_metrics.update(quant_stat_metrics)
            if forward_metrics:
                train_metrics.update(forward_metrics)
            if kd_enabled:
                train_metrics.update(
                    {
                        "training/task_loss": total_task_loss,
                        "training/kd_loss": total_kd_loss,
                    }
                )

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "monitor/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "loss_spike_meter": loss_spike_meter.state_dict(),
                        "forward_consistency_meter": forward_consistency_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "loss_spike_meter": loss_spike_meter.state_dict(),
                    "forward_consistency_meter": forward_consistency_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
        # Export a real (packed) tritonv2 quantized checkpoint for inference (rank0 only).
        try:
            try:
                from .export_tritonv2_quant import export_tritonv2_quantized_checkpoint
            except ImportError:
                from export_tritonv2_quant import export_tritonv2_quantized_checkpoint

            export_dst = os.path.join(args.train.save_checkpoint_path, "out")
            logger.info_rank0(
                "Exporting tritonv2 quantized checkpoint: src=%s -> dst=%s (bits=%s, group_size=%s, pack_dtype=%s)",
                hf_weights_path,
                export_dst,
                args.quantizer.n_bits,
                args.quantizer.group_size,
                "int32",
            )
            export_summary = export_tritonv2_quantized_checkpoint(
                src=hf_weights_path,
                dst=export_dst,
                bits=int(args.quantizer.n_bits),
                group_size=int(args.quantizer.group_size),
                pack_dtype="int32",
                weight_dtype="auto",
                exclude=[],
            )
            logger.info_rank0(
                "TritonV2 export done: converted=%d skipped=%d (config at %s)",
                len(export_summary.get("converted_modules", [])),
                len(export_summary.get("skipped_modules", [])),
                os.path.join(export_dst, "quantize_config.json"),
            )
        except Exception as e:
            logger.warning_rank0("TritonV2 export failed: %s", e)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
