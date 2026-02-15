import json
import math
import os
import shutil
import time
from contextlib import nullcontext
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
from veomni.distributed.pipeline import infer_pp_input_shape
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
    UniformRatioAssigner,
    ScoreProportionalRatioAssigner,
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


def _prepare_teacher_inputs(
    micro_batch: Dict[str, Any],
    enable_multisource: bool,
) -> Dict[str, torch.Tensor]:
    teacher_batch = {k: v for k, v in micro_batch.items()}
    if enable_multisource:
        teacher_batch.pop("ds_idx", None)
        teacher_batch.pop("source_name", None)
    teacher_inputs = {k: v for k, v in teacher_batch.items() if k != "labels"}
    return {
        k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in teacher_inputs.items()
    }


@dataclass
class _TeacherCudaGraphRunner:
    graph: "torch.cuda.CUDAGraph"
    static_inputs: Dict[str, torch.Tensor]
    static_outputs: Any
    input_keys: List[str]

    def matches(self, inputs: Dict[str, Any]) -> bool:
        if set(inputs.keys()) != set(self.input_keys):
            return False
        for key in self.input_keys:
            val = inputs.get(key)
            if not isinstance(val, torch.Tensor):
                return False
            static_val = self.static_inputs[key]
            if val.shape != static_val.shape or val.dtype != static_val.dtype or val.device != static_val.device:
                return False
        return True

    def replay(self, inputs: Dict[str, torch.Tensor]) -> Any:
        for key in self.input_keys:
            self.static_inputs[key].copy_(inputs[key], non_blocking=True)
        self.graph.replay()
        return self.static_outputs


def _teacher_cudagraph_skip_reason(
    teacher_inputs: Dict[str, Any],
    args: "Arguments",
) -> str | None:
    if get_device_type() != "cuda" or not torch.cuda.is_available():
        return "CUDA is not available"
    if args.train.rmpad:
        return "rmpad enabled (padding-free inputs)"
    if "cu_seqlens" in teacher_inputs or "max_seqlen" in teacher_inputs:
        return "padding-free inputs detected (cu_seqlens/max_seqlen)"
    if "input_ids" not in teacher_inputs:
        return "input_ids missing"
    input_ids = teacher_inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        return "input_ids is not a tensor"
    if input_ids.dim() != 2:
        return f"input_ids dim != 2 (got {input_ids.dim()})"
    if input_ids.shape[0] != args.train.micro_batch_size:
        return (
            f"input_ids batch={input_ids.shape[0]} != "
            f"micro_batch_size={args.train.micro_batch_size}"
        )
    ps = get_parallel_state()
    sp_size = ps.sp_size if ps.sp_enabled else 1
    max_seq_len = int(args.data.max_seq_len)
    padded_full_len = ((max_seq_len + sp_size - 1) // sp_size) * sp_size
    expected_local_len = padded_full_len // sp_size
    allowed_seq_lens = {max_seq_len, padded_full_len, expected_local_len}
    if input_ids.shape[1] not in allowed_seq_lens:
        return (
            f"input_ids seq_len={input_ids.shape[1]} not in expected {sorted(allowed_seq_lens)} "
            f"(max_seq_len={max_seq_len}, sp_size={sp_size})"
        )
    for key in ("attention_mask", "position_ids"):
        if key in teacher_inputs:
            val = teacher_inputs[key]
            if not isinstance(val, torch.Tensor):
                return f"{key} is not a tensor"
            if val.dim() == 2:
                expected_shapes = {
                    (input_ids.shape[0], expected_local_len),
                    (input_ids.shape[0], max_seq_len),
                    (input_ids.shape[0], padded_full_len),
                }
                if tuple(val.shape) not in expected_shapes:
                    return (
                        f"{key} shape {tuple(val.shape)} mismatched with expected "
                        f"{sorted(expected_shapes)}"
                    )
            elif val.dim() >= 3:
                if val.shape[-1] not in (expected_local_len, max_seq_len, padded_full_len):
                    return (
                        f"{key} last_dim={val.shape[-1]} mismatched with "
                        f"expected seq len {expected_local_len} or {max_seq_len} or {padded_full_len}"
                    )
    for key, val in teacher_inputs.items():
        if not isinstance(val, torch.Tensor):
            return f"non-tensor input detected: {key}"
    return None


def _build_teacher_cuda_graph(
    teacher_model: torch.nn.Module,
    teacher_inputs: Dict[str, torch.Tensor],
    args: "Arguments",
) -> _TeacherCudaGraphRunner | None:
    skip_reason = _teacher_cudagraph_skip_reason(teacher_inputs, args)
    if skip_reason:
        logger.warning_rank0("Skip teacher CUDA graph capture: %s", skip_reason)
        return None

    warmup_iters = max(0, int(getattr(args.distill, "teacher_cuda_graph_warmup_iters", 3)))
    static_inputs: Dict[str, torch.Tensor] = {
        key: torch.empty_like(val) for key, val in teacher_inputs.items()
    }
    for key in static_inputs:
        static_inputs[key].copy_(teacher_inputs[key], non_blocking=True)

    try:
        with torch.no_grad():
            for _ in range(warmup_iters):
                teacher_model(**static_inputs, use_cache=False)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_outputs = teacher_model(**static_inputs, use_cache=False)
    except Exception as exc:
        logger.warning_rank0("Teacher CUDA graph capture failed: %s", exc)
        return None

    return _TeacherCudaGraphRunner(
        graph=graph,
        static_inputs=static_inputs,
        static_outputs=static_outputs,
        input_keys=list(static_inputs.keys()),
    )


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


def _get_multistep_loss_weights(
    num_steps: int,
    weight_scheme: str,
    decay_factor: float = 0.9,
) -> List[float]:
    """Generate loss weights for multi-step KD based on the weighting scheme."""
    if num_steps <= 0:
        return []
    if weight_scheme == "decay":
        # Later steps have lower weight (exponential decay)
        weights = [decay_factor ** i for i in range(num_steps)]
    elif weight_scheme == "increase":
        # Later steps have higher weight (exponential increase)
        weights = [decay_factor ** (num_steps - 1 - i) for i in range(num_steps)]
    else:  # uniform
        weights = [1.0] * num_steps
    # Normalize weights to sum to 1
    total = sum(weights)
    return [w / total for w in weights]


def _compute_multistep_kd_loss_windowed(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    num_steps: int,
    stride: int,
    temperature: float,
    loss_weights: List[float],
) -> tuple[torch.Tensor | None, Dict[str, float]]:
    """
    Compute multi-step windowed KD loss on pre-computed logits.
    
    This is a sliding window approach that divides the sequence into multiple
    windows and computes KD loss for each window. This builds longer-range
    dependencies by ensuring the model learns to align with teacher across
    different positions in the sequence.
    
    Args:
        student_logits: Student model logits [batch, seq_len, vocab]
        teacher_logits: Teacher model logits [batch, seq_len, vocab]
        labels: Labels for loss masking [batch, seq_len]
        num_steps: Number of windows to compute KD loss
        stride: Size of each window (number of tokens)
        temperature: KD temperature
        loss_weights: Weight for each window's KD loss
        
    Returns:
        Tuple of (total_kd_loss, metrics_dict)
    """
    if student_logits is None or teacher_logits is None:
        return None, {}
    if student_logits.shape != teacher_logits.shape:
        return None, {}
    if num_steps <= 0 or stride <= 0:
        return None, {}
    
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Accumulate KD losses across windows
    total_kd_loss = None
    step_losses = []
    metrics = {}
    
    temperature = float(temperature)
    if temperature <= 0.0:
        temperature = 1.0
    
    # Compute KD loss for each window
    for step_idx in range(num_steps):
        window_start = step_idx * stride
        window_end = min((step_idx + 1) * stride, seq_len)
        
        if window_start >= seq_len:
            break
        
        # Extract logits for this window
        student_window = student_logits[:, window_start:window_end, :].float() / temperature
        teacher_window = teacher_logits[:, window_start:window_end, :].float() / temperature
        
        # Apply label mask if available
        if labels is not None:
            labels_window = labels[:, window_start:window_end].reshape(-1)
            mask = labels_window != -100
            if mask.any():
                student_window = student_window.reshape(-1, vocab_size)[mask]
                teacher_window = teacher_window.reshape(-1, vocab_size)[mask]
            else:
                # Skip this window if all labels are masked
                continue
        else:
            student_window = student_window.reshape(-1, vocab_size)
            teacher_window = teacher_window.reshape(-1, vocab_size)
        
        # Compute KD loss for this window
        window_kd_loss = F.kl_div(
            F.log_softmax(student_window, dim=-1),
            F.softmax(teacher_window, dim=-1),
            reduction="batchmean",
        )
        window_kd_loss = _to_local_tensor(window_kd_loss * (temperature ** 2))
        
        # Apply weight
        weight = loss_weights[step_idx] if step_idx < len(loss_weights) else loss_weights[-1]
        weighted_loss = weight * window_kd_loss
        step_losses.append(window_kd_loss.item())
        
        if total_kd_loss is None:
            total_kd_loss = weighted_loss
        else:
            total_kd_loss = total_kd_loss + weighted_loss
    
    # Collect metrics
    if step_losses:
        metrics["multistep_kd/num_windows"] = len(step_losses)
        metrics["multistep_kd/mean_window_loss"] = sum(step_losses) / len(step_losses)
        for i, loss_val in enumerate(step_losses):
            metrics[f"multistep_kd/window_{i}_loss"] = loss_val
    
    return total_kd_loss, metrics


def _teacher_generate_tokens(
    teacher_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_new_tokens: int,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher model generates new tokens autoregressively.
    
    Args:
        teacher_model: The FP teacher model
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        num_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding
        
    Returns:
        Tuple of (extended_input_ids, all_logits)
        - extended_input_ids: [batch, seq_len + num_new_tokens]
        - all_logits: [batch, num_new_tokens, vocab_size] - logits for each generated position
    """
    batch_size, orig_seq_len = input_ids.shape
    device = input_ids.device
    
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone() if attention_mask is not None else None
    
    all_logits = []
    
    with torch.no_grad():
        for step in range(num_new_tokens):
            # Forward pass
            outputs = teacher_model(
                input_ids=current_ids,
                attention_mask=current_mask,
                use_cache=False,
            )
            
            # Get logits for the last position
            next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
            all_logits.append(next_token_logits)
            
            # Generate next token
            if do_sample and temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if current_mask is not None:
                new_mask = torch.ones(batch_size, 1, device=device, dtype=current_mask.dtype)
                current_mask = torch.cat([current_mask, new_mask], dim=1)
    
    # Stack all logits: [batch, num_new_tokens, vocab]
    all_logits = torch.stack(all_logits, dim=1)
    
    return current_ids, all_logits


def _compute_multistep_kd_loss_extrapolate(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    labels: torch.Tensor | None,
    num_steps: int,
    stride: int,
    temperature: float,
    loss_weights: List[float],
    model_fwd_context,
    sp_group: "dist.ProcessGroup | None" = None,
) -> tuple[torch.Tensor | None, Dict[str, float]]:
    """
    Compute multi-step extrapolation KD loss.
    
    The teacher model generates new tokens beyond the original sequence,
    then the student model learns to align with teacher's predictions
    on this extended sequence. This builds long-range dependencies.
    
    Flow:
    1. Teacher generates num_steps * stride new tokens
    2. Student does forward on the extended sequence (original + generated)
    3. Compute KD loss between student and teacher logits on generated positions
    
    Args:
        student_model: The quantized student model
        teacher_model: The FP teacher model  
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        position_ids: Position IDs [batch, seq_len]
        labels: Labels (used for original sequence loss)
        num_steps: Number of generation steps
        stride: Tokens to generate per step
        temperature: KD temperature
        loss_weights: Weight for each step's KD loss
        model_fwd_context: Context manager for model forward
        sp_group: Sequence parallel process group (for gather/scatter)
        
    Returns:
        Tuple of (total_kd_loss, metrics_dict)
    """
    if num_steps <= 0 or stride <= 0:
        return None, {}
    
    batch_size, orig_seq_len = input_ids.shape
    device = input_ids.device
    num_new_tokens = num_steps * stride
    
    metrics = {}
    
    # Handle sequence parallel: gather full sequence to rank 0
    # For simplicity, we'll do generation on the full sequence
    # In SP mode, each rank has a shard of the sequence
    gathered_input_ids = input_ids
    gathered_attention_mask = attention_mask
    
    if sp_group is not None and dist.get_world_size(sp_group) > 1:
        # Gather input_ids across SP ranks
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        
        # All-gather input_ids
        gathered_list = [torch.zeros_like(input_ids) for _ in range(sp_size)]
        dist.all_gather(gathered_list, input_ids, group=sp_group)
        gathered_input_ids = torch.cat(gathered_list, dim=1)  # [batch, full_seq_len]
        
        if attention_mask is not None:
            mask_list = [torch.zeros_like(attention_mask) for _ in range(sp_size)]
            dist.all_gather(mask_list, attention_mask, group=sp_group)
            gathered_attention_mask = torch.cat(mask_list, dim=1)
    
    # Step 1: Teacher generates new tokens
    extended_ids, teacher_gen_logits = _teacher_generate_tokens(
        teacher_model=teacher_model,
        input_ids=gathered_input_ids,
        attention_mask=gathered_attention_mask,
        num_new_tokens=num_new_tokens,
        temperature=1.0,  # Use greedy for generation
        do_sample=False,
    )
    # extended_ids: [batch, orig_seq_len + num_new_tokens]
    # teacher_gen_logits: [batch, num_new_tokens, vocab]
    
    metrics["multistep_kd/num_generated_tokens"] = num_new_tokens
    
    # Step 2: Teacher forward on extended sequence to get full logits
    with torch.no_grad():
        teacher_out = teacher_model(
            input_ids=extended_ids,
            attention_mask=torch.ones(batch_size, extended_ids.shape[1], device=device) 
                if gathered_attention_mask is not None else None,
            use_cache=False,
        )
        teacher_full_logits = teacher_out.logits  # [batch, orig_seq_len + num_new_tokens, vocab]
    
    # Step 3: Student forward on extended sequence
    # Prepare extended attention mask and position ids
    extended_seq_len = extended_ids.shape[1]
    extended_attention_mask = None
    if gathered_attention_mask is not None:
        extended_attention_mask = torch.ones(batch_size, extended_seq_len, device=device, dtype=gathered_attention_mask.dtype)
    
    extended_position_ids = None
    if position_ids is not None:
        # Extend position ids
        if sp_group is not None and dist.get_world_size(sp_group) > 1:
            # Gather position_ids
            sp_size = dist.get_world_size(sp_group)
            pos_list = [torch.zeros_like(position_ids) for _ in range(sp_size)]
            dist.all_gather(pos_list, position_ids, group=sp_group)
            gathered_position_ids = torch.cat(pos_list, dim=1)
        else:
            gathered_position_ids = position_ids
        
        # Extend with new positions
        last_pos = gathered_position_ids[:, -1:] + 1
        new_positions = last_pos + torch.arange(num_new_tokens, device=device).unsqueeze(0)
        extended_position_ids = torch.cat([gathered_position_ids, new_positions], dim=1)
    
    # Student forward
    with model_fwd_context:
        student_out = student_model(
            input_ids=extended_ids,
            attention_mask=extended_attention_mask,
            position_ids=extended_position_ids,
            use_cache=False,
        )
        student_full_logits = student_out.logits  # [batch, orig_seq_len + num_new_tokens, vocab]
    
    # Step 4: Compute KD loss on generated positions (step by step)
    total_kd_loss = None
    step_losses = []
    
    full_orig_seq_len = gathered_input_ids.shape[1]  # Full sequence length before generation
    
    for step_idx in range(num_steps):
        # Position range for this step's generated tokens
        # Logits at position i predict token at position i+1
        # So for generated tokens at positions [full_orig_seq_len + step_idx*stride : full_orig_seq_len + (step_idx+1)*stride]
        # We need logits at positions [full_orig_seq_len + step_idx*stride - 1 : full_orig_seq_len + (step_idx+1)*stride - 1]
        logit_start = full_orig_seq_len + step_idx * stride - 1
        logit_end = full_orig_seq_len + (step_idx + 1) * stride - 1
        
        if logit_start < 0:
            logit_start = 0
        if logit_end > student_full_logits.shape[1]:
            logit_end = student_full_logits.shape[1]
        
        if logit_start >= logit_end:
            break
        
        student_step_logits = student_full_logits[:, logit_start:logit_end, :]
        teacher_step_logits = teacher_full_logits[:, logit_start:logit_end, :]
        
        # Compute KD loss for this step
        step_kd_loss = _compute_kd_loss(
            student_logits=student_step_logits,
            teacher_logits=teacher_step_logits,
            labels=None,  # No labels for generated tokens
            temperature=temperature,
        )
        
        if step_kd_loss is not None:
            weight = loss_weights[step_idx] if step_idx < len(loss_weights) else loss_weights[-1]
            weighted_loss = weight * step_kd_loss
            step_losses.append(step_kd_loss.item())
            # Force sync each multistep KD window to surface async CUDA errors earlier.
            torch.cuda.synchronize()
            
            if total_kd_loss is None:
                total_kd_loss = weighted_loss
            else:
                total_kd_loss = total_kd_loss + weighted_loss
    
    # Collect metrics
    if step_losses:
        metrics["multistep_kd/num_steps"] = len(step_losses)
        metrics["multistep_kd/mean_step_loss"] = sum(step_losses) / len(step_losses)
        for i, loss_val in enumerate(step_losses):
            metrics[f"multistep_kd/step_{i}_loss"] = loss_val
    
    return total_kd_loss, metrics


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


def _normalize_profiler_activities(activities: List[str]) -> List["torch.profiler.ProfilerActivity"]:
    mapping = {
        "CPU": torch.profiler.ProfilerActivity.CPU,
        "CUDA": torch.profiler.ProfilerActivity.CUDA,
    }
    resolved: List[torch.profiler.ProfilerActivity] = []
    for name in activities:
        key = str(name).strip().upper()
        if key in mapping:
            if mapping[key] == torch.profiler.ProfilerActivity.CUDA and not torch.cuda.is_available():
                continue
            resolved.append(mapping[key])
    if not resolved:
        resolved = [torch.profiler.ProfilerActivity.CPU]
    # de-dup while preserving order
    seen = set()
    uniq: List[torch.profiler.ProfilerActivity] = []
    for act in resolved:
        if act in seen:
            continue
        uniq.append(act)
        seen.add(act)
    return uniq


def _unwrap_model_for_profiling(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _install_module_profiler_hooks(model: torch.nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _pre_hook(module, _inputs):
        stack = module.__dict__.setdefault("_profiler_ctx_stack", [])
        name = module.__dict__.get("_profiler_qualname", module.__class__.__name__)
        ctx = torch.profiler.record_function(name)
        ctx.__enter__()
        stack.append(ctx)

    def _post_hook(module, _inputs, _output):
        stack = module.__dict__.get("_profiler_ctx_stack")
        if stack:
            ctx = stack.pop()
            ctx.__exit__(None, None, None)

    for name, module in model.named_modules():
        qualname = f"module:{name}:{module.__class__.__name__}" if name else f"module:<root>:{module.__class__.__name__}"
        module.__dict__["_profiler_qualname"] = qualname
        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))

    return handles


def _compute_quant_stats_for_module(
    module: IntQuantLinear,
    percentiles: List[float],
    prev_scale: torch.Tensor | None,
    prev_weight: torch.Tensor | None,
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor] | None:
    q = getattr(module, "weight_quantizer", None)
    if (
        q is None
        or not hasattr(q, "scale")
        or not hasattr(q, "cal_qparams")
        or not hasattr(q, "_quantize")
        or not hasattr(q, "_dequantize")
    ):
        return None
    group_size = getattr(q, "group_size", None)
    if group_size is None or int(group_size) <= 0:
        return None

    with torch.no_grad():
        weight = _to_local_tensor(module.weight.detach())
        scale = _to_local_tensor(q.scale.detach())
        zp = getattr(q, "zero_point", None)
        if isinstance(zp, torch.Tensor):
            zp = _to_local_tensor(zp.detach())
        else:
            zp = None
        scale, round_zero_point = q.cal_qparams(scale, zp)
        x = weight.reshape(-1, int(group_size))
        scale = scale.reshape(-1, 1)
        if isinstance(round_zero_point, torch.Tensor):
            round_zero_point = round_zero_point.reshape(-1, 1)
        else:
            round_zero_point = None

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
            if round_zero_point is not None:
                round_zero_point = round_zero_point[mask]

        x_int = q._quantize(x, scale, round_zero_point)
        x_dequant = q._dequantize(x_int, scale, round_zero_point)

        qmin = getattr(q, "qmin", None)
        qmax = getattr(q, "qmax", None)
        if qmin is not None and qmax is not None:
            sat_mask = (x_int <= qmin) | (x_int >= qmax)
            sat_rate = sat_mask.float().mean().item()
        else:
            sat_rate = 0.0

        abs_x = x.abs()
        abs_max = abs_x.max().item()
        clip_rate = sat_rate

        stats: Dict[str, float] = {
            "clip_rate": clip_rate,
            "saturation_rate": sat_rate,
            "weight_abs_max": abs_max,
        }
        for p in percentiles:
            if 0.0 < p < 1.0:
                stats[f"weight_abs_{_percentile_label(p)}"] = torch.quantile(abs_x, p).item()

        weight_flat = x.reshape(-1).float()
        quant_residual = (x_dequant - x).reshape(-1).float()
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
    enable_teacher_cuda_graph: bool = field(
        default=False,
        metadata={"help": "Enable CUDA graph replay for teacher forward (fixed shapes only)."},
    )
    teacher_cuda_graph_warmup_iters: int = field(
        default=3,
        metadata={"help": "Warmup iterations before capturing teacher CUDA graph."},
    )
    # Multi-step distillation arguments
    enable_multistep_kd: bool = field(
        default=False,
        metadata={"help": "Enable multi-step autoregressive distillation for long-range dependency."},
    )
    multistep_kd_steps: int = field(
        default=4,
        metadata={"help": "Number of autoregressive steps for multi-step KD."},
    )
    multistep_kd_stride: int = field(
        default=1,
        metadata={"help": "Token stride per autoregressive step (how many tokens to generate per step)."},
    )
    multistep_kd_loss_weight: str = field(
        default="uniform",
        metadata={"help": "Loss weighting scheme for multi-step KD: uniform | decay | increase."},
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
        symmetric=getattr(args.quantizer, "symmetric", False),
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
        if args.quantizer.quant_type == "seq2bit":
            logger.warning_rank0(
                "Seq2BitQuantizer currently uses torch simulated packing/export path; skip IntQuantLinearInfra conversion."
            )
        else:
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
    pp_input_shape = None
    if args.train.pipeline_parallel_size > 1:
        pp_input_shape = infer_pp_input_shape(
            model,
            micro_batch_size=args.train.micro_batch_size,
            max_seq_len=args.data.max_seq_len,
            tp_size=args.train.tensor_parallel_size,
        )
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
        pp_input_shape=pp_input_shape,
    )
    
    # Substitute HF flash attention with ring attention if CP is enabled
    ps = get_parallel_state()
    if ps.cp_enabled:
        from veomni.distributed.sequence_parallel import (
            is_ring_flash_attn_available,
            substitute_hf_ring_attn,
        )
        if is_ring_flash_attn_available():
            # Get number of KV heads for GQA stride
            num_kv_heads = getattr(model_config, "num_key_value_heads", None)
            num_heads = getattr(model_config, "num_attention_heads", None)
            heads_k_stride = 1
            if num_kv_heads is not None and num_heads is not None and num_kv_heads < num_heads:
                heads_k_stride = num_heads // num_kv_heads
            # ring-flash-attn requires nheads_k % heads_k_stride == 0
            if num_kv_heads is not None and heads_k_stride > 1:
                if num_kv_heads % heads_k_stride != 0:
                    logger.warning_rank0(
                        "Skip ring attention: incompatible GQA heads for ring-flash-attn "
                        f"(num_kv_heads={num_kv_heads}, num_heads={num_heads}, "
                        f"heads_k_stride={heads_k_stride})."
                    )
                else:
                    substitute_hf_ring_attn(heads_k_stride=heads_k_stride)
                    logger.info_rank0(
                        f"Ring attention enabled for CP (cp_size={ps.cp_size}, heads_k_stride={heads_k_stride})"
                    )
            else:
                substitute_hf_ring_attn(heads_k_stride=heads_k_stride)
                logger.info_rank0(
                    f"Ring attention enabled for CP (cp_size={ps.cp_size}, heads_k_stride={heads_k_stride})"
                )
        else:
            logger.warning_rank0(
                "CP enabled but ring-flash-attn not available. "
                "Install with: pip install ring-flash-attn"
            )
    
    reinit_quant_params(model)
    sanitize_quant_params(model)
    
    # <------QAT prefetch configuration-------->
    if args.train.num_to_forward_prefetch > 0 or args.train.num_to_backward_prefetch > 0:
        logger.info_rank0(
            "Configuring manual prefetching: forward=%d, backward=%d",
            args.train.num_to_forward_prefetch,
            args.train.num_to_backward_prefetch,
        )
        layers = getattr(model, "layers", None)
        if layers is None and hasattr(model, "model"):
            layers = getattr(model.model, "layers", None)
            
        if layers is not None:
            num_fwd = args.train.num_to_forward_prefetch
            if num_fwd > 0:
                for i, layer in enumerate(layers):
                    if i >= len(layers) - 1: # No next layer
                        continue
                        
                    # We want to prefetch up to num_fwd next layers
                    targets = layers[i + 1 : i + 1 + num_fwd]
                    
                    prefetch_modules = []
                    for t in targets:
                        # Use _fsdp_modules if available (populated by FSDP2 setup)
                        if hasattr(t, "_fsdp_modules"):
                            prefetch_modules.extend(reversed(t._fsdp_modules))
                        else:
                            prefetch_modules.append(t)
                    
                    if hasattr(layer, "set_modules_to_forward_prefetch"):
                        layer.set_modules_to_forward_prefetch(prefetch_modules)

            num_bwd = args.train.num_to_backward_prefetch
            if num_bwd > 0:
                for i, layer in enumerate(layers):
                    if i == 0:
                        continue
                    
                    # Prefetch previous layers: i-1, i-2... (in reverse order of distance)
                    start_idx = max(0, i - num_bwd)
                    targets = layers[start_idx : i]
                    targets = list(reversed(targets))
                    
                    prefetch_modules = []
                    for t in targets:
                        if hasattr(t, "_fsdp_modules"):
                             prefetch_modules.extend(reversed(t._fsdp_modules))
                        else:
                             prefetch_modules.append(t)

                    if hasattr(layer, "set_modules_to_backward_prefetch"):
                        layer.set_modules_to_backward_prefetch(prefetch_modules)
        else:
             logger.warning_rank0("Could not find model.layers to configure manual prefetching.")

    # Gradual sync only for non-infra (since infra has no quantizer object)
    if not getattr(args.quantizer, "enable_infra", False):
        _sync_gradual_quantizer_metadata(model)
        logger.info_rank0("Reinitialized quantizer params from loaded weights.")
    else:
        logger.info_rank0("Skipping gradual metadata sync for Infra mode.")

    kd_mode = args.distill.kd_mode
    kd_enabled = kd_mode != "none"
    if kd_enabled and get_parallel_state().pp_enabled:
        raise NotImplementedError("KD is not supported with pipeline parallelism yet.")
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

    # Multi-step KD configuration
    enable_multistep_kd = args.distill.enable_multistep_kd and kd_enabled
    multistep_kd_steps = int(args.distill.multistep_kd_steps)
    multistep_kd_stride = int(args.distill.multistep_kd_stride)
    multistep_kd_loss_weights = []
    if enable_multistep_kd:
        if multistep_kd_steps <= 0:
            logger.warning_rank0("multistep_kd_steps=%s <= 0; disabling multi-step KD.", multistep_kd_steps)
            enable_multistep_kd = False
        elif multistep_kd_stride <= 0:
            logger.warning_rank0("multistep_kd_stride=%s <= 0; disabling multi-step KD.", multistep_kd_stride)
            enable_multistep_kd = False
        else:
            multistep_kd_loss_weights = _get_multistep_loss_weights(
                num_steps=multistep_kd_steps,
                weight_scheme=args.distill.multistep_kd_loss_weight,
            )
            logger.info_rank0(
                "Multi-step KD enabled: steps=%d, stride=%d, weight_scheme=%s, weights=%s",
                multistep_kd_steps,
                multistep_kd_stride,
                args.distill.multistep_kd_loss_weight,
                [f"{w:.4f}" for w in multistep_kd_loss_weights],
            )

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
            torch_dtype="bfloat16",
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
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            basic_modules=teacher_model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
            pp_input_shape=pp_input_shape,
        )
        logger.info_rank0("KD teacher model forced to bfloat16 to reduce memory usage.")
        teacher_model.eval()
        teacher_model.requires_grad_(False)
    teacher_graph = None
    teacher_graph_failed = False
    if kd_enabled and args.distill.enable_teacher_cuda_graph:
        if get_device_type() != "cuda" or not torch.cuda.is_available():
            logger.warning_rank0("Teacher CUDA graph requested but CUDA is unavailable; disabling.")
            teacher_graph_failed = True
        else:
            ps = get_parallel_state()
            sp_size = ps.sp_size if ps.sp_enabled else 1
            max_seq_len = int(args.data.max_seq_len)
            padded_full_len = ((max_seq_len + sp_size - 1) // sp_size) * sp_size
            expected_local_len = padded_full_len // sp_size
            logger.info_rank0(
                "Teacher CUDA graph enabled (fixed shapes): micro_batch_size=%s, "
                "max_seq_len=%s, sp_size=%s, local_seq_len=%s, warmup_iters=%s",
                args.train.micro_batch_size,
                max_seq_len,
                sp_size,
                expected_local_len,
                args.distill.teacher_cuda_graph_warmup_iters,
            )
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

    if args.train.enable_profiling and args.train.debug:
        logger.warning_rank0(
            "Both enable_profiling and debug profiling are enabled. Expect higher overhead."
        )

    debug_profiler = None
    module_profiler_hooks: List[torch.utils.hooks.RemovableHandle] = []
    if args.train.debug and args.train.debug_profile_this_rank:
        rank = dist.get_rank() if dist.is_initialized() else args.train.global_rank
        trace_root = os.path.join(args.train.output_dir, "debug_profiler")
        trace_dir = os.path.join(trace_root, f"rank{rank}")
        os.makedirs(trace_dir, exist_ok=True)
        activities = _normalize_profiler_activities(args.train.debug_profiler_activities)
        wait = max(0, int(args.train.debug_profiler_wait))
        warmup = max(0, int(args.train.debug_profiler_warmup))
        active = max(1, int(args.train.debug_profiler_active))
        repeat = max(1, int(args.train.debug_profiler_repeat))
        schedule = torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        def _debug_trace_handler(prof):
            trace_path = os.path.join(trace_dir, f"trace_step{prof.step_num}.json")
            prof.export_chrome_trace(trace_path)

        debug_profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=_debug_trace_handler,
            record_shapes=args.train.debug_profiler_record_shapes,
            profile_memory=args.train.debug_profiler_profile_memory,
            with_stack=args.train.debug_profiler_with_stack,
        )
        debug_profiler.start()
        module_profiler_hooks = _install_module_profiler_hooks(_unwrap_model_for_profiling(model))

    def _record_function(name: str):
        if debug_profiler is None:
            return nullcontext()
        return torch.profiler.record_function(name)

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
    # gradual 量化：全程 use_weight_quant=True，通过 group_mask 或 group_ratio 控制量化强度
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
        priority_kind = priority_type
        if priority_type == "uniform":
            priority_calculator = UniformPriorityCalculator()
        elif priority_type == "magnitude":
            priority_calculator = MagnitudePriorityCalculator()
        else:
            logger.info_rank0(f"Unknown priority_type={priority_type}, fallback to uniform")
            priority_calculator = UniformPriorityCalculator()
            priority_kind = "uniform"

        # 软调度：沿用原有 priority_type 来决定 ratio 分配策略
        if priority_kind == "uniform":
            ratio_assigner = UniformRatioAssigner()
        else:
            ratio_assigner = ScoreProportionalRatioAssigner()

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
            ratio_assigner=ratio_assigner,
        )
        gradual_controller = GradualQuantController(model, scheduler)

        # Gradual quantization: quantizer is always enabled; group_mask/group_ratio 控制量化强度
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
        if global_step == 50:
            helper.empty_cache()
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
        pending_micro_batches = None
        if (
            kd_enabled
            and args.distill.enable_teacher_cuda_graph
            and teacher_model is not None
            and teacher_graph is None
            and not teacher_graph_failed
        ):
            try:
                pending_micro_batches = next(data_iterator)
                warmup_batch = {k: v for k, v in pending_micro_batches[0].items()}
                teacher_inputs = _prepare_teacher_inputs(warmup_batch, args.data.enable_multisource)
                teacher_graph = _build_teacher_cuda_graph(
                    teacher_model=teacher_model,
                    teacher_inputs=teacher_inputs,
                    args=args,
                )
                if teacher_graph is None:
                    teacher_graph_failed = True
                else:
                    logger.info_rank0("Teacher CUDA graph captured and warmed up.")
            except StopIteration:
                logger.warning_rank0("No batch available for teacher CUDA graph warmup; disabling.")
                teacher_graph_failed = True
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                with _record_function("data_load"):
                    if pending_micro_batches is not None:
                        micro_batches = pending_micro_batches
                        pending_micro_batches = None
                    else:
                        micro_batches = next(data_iterator)
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
            accumulated_multistep_kd_metrics: Dict[str, float] = {}
            debug_batch = None
            synchronize()
            start_time = time.time()
            if get_parallel_state().pp_enabled:
                pipeline_model = model.module if hasattr(model, "module") else model
                if not hasattr(pipeline_model, "forward_backward_1f1b"):
                    raise RuntimeError("Pipeline model does not implement forward_backward_1f1b.")
                if teacher_model is not None:
                    raise NotImplementedError("KD is not supported with pipeline parallelism yet.")

                prepared_micro_batches = []
                for micro_batch in micro_batches:
                    environ_meter.add(micro_batch)
                    if args.data.enable_multisource:
                        micro_batch.pop("ds_idx", None)
                        micro_batch.pop("source_name", None)

                    if debug_batch is None:
                        debug_batch = {k: v for k, v in micro_batch.items()}
                    prepared_micro_batches.append(micro_batch)

                def loss_fn(output, micro_batch):
                    nonlocal total_task_loss, total_qweight_l2_reg, total_qweight_l2_scaled
                    task_loss = pipeline_model._compute_lm_loss(output, micro_batch)
                    task_loss = _to_local_tensor(task_loss)
                    combined_loss = task_loss
                    if enable_qweight_l2_reg:
                        reg_loss = _compute_weight_l2_reg_loss(pipeline_model)
                        if reg_loss is not None:
                            combined_loss = combined_loss + qweight_l2_reg_lambda * reg_loss
                            reg_loss_val = float(reg_loss.detach())
                            total_qweight_l2_reg += reg_loss_val / len(micro_batches)
                            total_qweight_l2_scaled += (
                                qweight_l2_reg_lambda * reg_loss_val / len(micro_batches)
                            )
                    total_task_loss += task_loss.item() / len(micro_batches)
                    return combined_loss

                with _record_function("pipeline_fwd_bwd"):
                    total_loss = pipeline_model.forward_backward_1f1b(
                        prepared_micro_batches,
                        model_fwd_context=model_fwd_context,
                        model_bwd_context=model_bwd_context,
                        use_cache=False,
                        loss_fn=loss_fn,
                    )
            else:
                for micro_batch in micro_batches:
                    environ_meter.add(micro_batch)
                    if args.data.enable_multisource:
                        micro_batch.pop("ds_idx", None)
                        micro_batch.pop("source_name", None)

                    if debug_batch is None:
                        debug_batch = {k: v for k, v in micro_batch.items()}
                    # For ring-flash-attn with CP, update cu_seqlens before each forward.
                    ps = get_parallel_state()
                    if ps.cp_enabled:
                        try:
                            from veomni.distributed.sequence_parallel import (
                                is_ring_flash_attn_available,
                                update_ring_attn_cu_seqlens,
                            )
                            if is_ring_flash_attn_available():
                                cu_seqlens = micro_batch.get("cu_seq_lens_q")
                                if cu_seqlens is not None:
                                    update_ring_attn_cu_seqlens(cu_seqlens)
                        except Exception:
                            # Avoid breaking training if ring-flash-attn is unavailable.
                            pass
                    micro_batch = {
                        k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in micro_batch.items()
                    }
                    with _record_function("forward"):
                        with model_fwd_context:
                            student_out = model(**micro_batch, use_cache=False)
                            task_loss = student_out.loss.mean()
                            task_loss = _to_local_tensor(task_loss)
                            combined_loss = task_loss
                            kd_loss = None
                            multistep_kd_metrics = {}
                            if teacher_model is not None:
                                if enable_multistep_kd:
                                    # Multi-step extrapolation KD: teacher generates, student aligns
                                    ps = get_parallel_state()
                                    sp_group = ps.sp_group if ps.sp_enabled else None
                                    kd_loss, multistep_kd_metrics = _compute_multistep_kd_loss_extrapolate(
                                        student_model=model,
                                        teacher_model=teacher_model,
                                        input_ids=micro_batch["input_ids"],
                                        attention_mask=micro_batch.get("attention_mask"),
                                        position_ids=micro_batch.get("position_ids"),
                                        labels=micro_batch.get("labels"),
                                        num_steps=multistep_kd_steps,
                                        stride=multistep_kd_stride,
                                        temperature=kd_temperature,
                                        loss_weights=multistep_kd_loss_weights,
                                        model_fwd_context=model_fwd_context,
                                        sp_group=sp_group,
                                    )
                                else:
                                    # Single-step KD (original behavior)
                                    teacher_inputs = _prepare_teacher_inputs(
                                        micro_batch, args.data.enable_multisource
                                    )
                                    with torch.no_grad():
                                        if teacher_graph is not None:
                                            if not teacher_graph.matches(teacher_inputs):
                                                raise RuntimeError(
                                                    "Teacher CUDA graph input mismatch; "
                                                    "check max_seq_len/sp_size or disable graph."
                                                )
                                            teacher_out = teacher_graph.replay(teacher_inputs)
                                        else:
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
                                        getattr(teacher_out, "logits", None) if not enable_multistep_kd else None,
                                        micro_batch.get("labels"),
                                    )
                                    student_keys = _maybe_get_output_keys(student_out)
                                    kd_skip_msg = (
                                        f"KD enabled but {kd_skip_reason}; skipping KD loss."
                                        if kd_skip_reason
                                        else "KD enabled but logits missing or mismatched; skipping KD loss."
                                    )
                                    if student_keys:
                                        kd_skip_msg = f"{kd_skip_msg} student_out.keys={student_keys}"
                                    logger.warning_rank0(kd_skip_msg)
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

                    with _record_function("backward"):
                        with model_bwd_context:
                            loss.backward()

                    total_loss += loss.item()
                    total_task_loss += task_loss.item() / len(micro_batches)
                    if kd_loss is not None:
                        total_kd_loss += kd_loss.item() / len(micro_batches)
                    # Accumulate multi-step KD metrics
                    if multistep_kd_metrics:
                        for key, val in multistep_kd_metrics.items():
                            if key in accumulated_multistep_kd_metrics:
                                accumulated_multistep_kd_metrics[key] += val / len(micro_batches)
                            else:
                                accumulated_multistep_kd_metrics[key] = val / len(micro_batches)
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

            with _record_function("optimizer_step"):
                optimizer.step()

                # Record event for FSDP2 optimization
                if hasattr(model, "set_post_optim_event"):
                    evt = torch.cuda.Event()
                    evt.record()
                    model.set_post_optim_event(evt)
                
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
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
            # Update gradual quantization state: group_mask/group_ratio controls quantization strength
            # 在 gradual 模式下，全程 use_weight_quant=True，但通过 group_mask/group_ratio 控制量化强度
            if gradual_controller is not None:
                gradual_controller.on_step_end(step=global_step, epoch=epoch)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # Collect metrics across data-parallel ranks only.
            # Loss may already be reduced inside sequence-parallel group in model loss functions.
            # Averaging again on fsdp_group (dp_sp) can over-divide by sp_size.
            metric_group = get_parallel_state().dp_group
            total_loss, grad_norm, total_task_loss, total_kd_loss = all_reduce(
                (total_loss, grad_norm, total_task_loss, total_kd_loss),
                group=metric_group,
            )
            qweight_metrics: Dict[str, float] = {}
            if enable_qweight_l2_reg:
                qweight_l2_reg, qweight_l2_scaled = all_reduce(
                    (total_qweight_l2_reg, total_qweight_l2_scaled),
                    group=metric_group,
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
                # Add multi-step KD metrics if available
                if accumulated_multistep_kd_metrics:
                    train_metrics.update(accumulated_multistep_kd_metrics)

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

            if debug_profiler is not None:
                debug_profiler.step()

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

    if debug_profiler is not None:
        debug_profiler.stop()
    for handle in module_profiler_hooks:
        handle.remove()

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

            export_root = args.train.save_checkpoint_path
            export_dst = os.path.join(export_root, "out")
            export_dst_dequant = os.path.join(export_root, "out_dequant")
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
                dst_dequant=export_dst_dequant,
                save_dequant=True,
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
            if os.path.isdir(export_dst_dequant):
                logger.info_rank0("Dequant checkpoint saved at: %s", export_dst_dequant)
        except Exception as e:
            logger.warning_rank0("TritonV2 export failed: %s", e)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
