import os
import re
import sys
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from easydict import EasyDict
from tqdm import tqdm

import accelerate
import yaml

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------- Repo path / env ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT_TASKS_PATH = REPO_ROOT / "VeOmni" / "tasks" / "quantize"
if str(QUANT_TASKS_PATH) not in sys.path:
    sys.path.append(str(QUANT_TASKS_PATH))

HF_HOME = REPO_ROOT / "hf_home"
os.environ.setdefault("HF_HOME", str(HF_HOME))

print("HF_HOME:", HF_HOME)

# Your local imports
import load_tritonv2_quant  # noqa: E402
from EfficientQAT.datautils_block import get_loaders  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402


# ---------------- Distributed helpers ----------------
@dataclass
class DistConfig:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    tp_mesh: Any = None


def init_distributed(tp_size: int) -> DistConfig:
    if tp_size <= 1:
        return DistConfig()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "tp_size > 1 requires torchrun. Example:\n"
            "  torchrun --nproc_per_node=4 test/eval_batch.py --tp-size 4"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != tp_size:
        raise RuntimeError(f"tp_size={tp_size} but WORLD_SIZE={world_size}")

    tp_mesh = None
    try:
        from torch.distributed.device_mesh import init_device_mesh

        tp_mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
    except Exception:
        tp_mesh = dist.group.WORLD

    return DistConfig(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank, tp_mesh=tp_mesh)


def rank0_print(enabled: bool, *args, **kwargs):
    if enabled:
        print(*args, **kwargs)


def apply_tp_for_eval(model: torch.nn.Module, tp_mesh) -> None:
    from EfficientQAT.core.linear.q_linear_pack import PackableQuantLinear

    for name, m in model.named_modules():
        if not isinstance(m, PackableQuantLinear):
            continue
        if name.endswith("lm_head") or name.endswith(".lm_head"):
            continue
        try:
            if name.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                m.set_tp_mesh(tp_mesh, tp_mode="col", tp_dim="tp", gather_output=True)
            elif name.endswith(("o_proj", "down_proj")):
                m.set_tp_mesh(tp_mesh, tp_mode="row", tp_dim="tp", input_is_parallel=False)
            else:
                # default to safe full output to avoid shape mismatches
                m.set_tp_mesh(tp_mesh, tp_mode="col", tp_dim="tp", gather_output=True)
        except Exception as exc:
            print(f"[WARN] TP skip {name}: {exc}")


# ---------------- Parallel config ----------------
@dataclass
class ParallelConfig:
    tp_size: int = 1
    max_mem_ratio: float = 0.85
    compile_model: bool = False


def get_free_mem_gib(device_id: int) -> float:
    free, total = torch.cuda.mem_get_info(device_id)
    return float(total) / (1 << 30)


def format_max_memory(device_ids: List[int], ratio: float) -> Dict[int, str]:
    mmap: Dict[int, str] = {}
    for d in device_ids:
        # conservative: use total*ratio; you may tune ratio down if OOM
        total_gib = get_free_mem_gib(d)
        mmap[d] = f"{total_gib * ratio:.0f}GiB"
    return mmap


def pick_tp_devices(tp_size: int) -> List[int]:
    """
    TP-only: use the first tp_size visible GPUs.
    """
    avail = torch.cuda.device_count()
    if avail < tp_size:
        raise RuntimeError(f"Need {tp_size} GPUs for tp, but only {avail} visible.")
    return list(range(tp_size))


def set_primary_device(device_id: int):
    torch.cuda.set_device(device_id)


def set_eval_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Quant config helpers (your original logic) ----------------
def _should_skip_module(module_name: str, skip_names) -> bool:
    return any(module_name == skip or module_name.endswith(f".{skip}") for skip in skip_names)


def _convert_linear_with_skip(module: torch.nn.Module, prefix: str, config, skip_names) -> None:
    from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if _should_skip_module(child_prefix, skip_names):
            continue
        if isinstance(child, torch.nn.Linear) and not isinstance(child, IntQuantLinear):
            setattr(module, name, IntQuantLinear.from_float(child_prefix, child, config))
        else:
            _convert_linear_with_skip(child, child_prefix, config, skip_names)


def _find_veomni_cli_yaml(quant_path: str):
    for parent in Path(quant_path).resolve().parents:
        candidate = parent / "veomni_cli.yaml"
        if candidate.is_file():
            return candidate
    return None


def _load_quant_params_from_cli(quant_path: str) -> dict:
    cli_path = _find_veomni_cli_yaml(quant_path)
    if cli_path is None:
        return {}
    with open(cli_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("quantizer", {})


def _infer_qat_config_from_path(quant_path: str):
    quant_type = "uniform_affine"
    bits = None
    group_size = None

    cli_cfg = _load_quant_params_from_cli(quant_path)
    if cli_cfg:
        quant_type = cli_cfg.get("quant_type", quant_type)
        bits = cli_cfg.get("n_bits") or bits
        group_size = cli_cfg.get("group_size") or group_size

    m = re.search(r"w(\d+)g(\d+)", quant_path)
    if m:
        bits = bits or int(m.group(1))
        group_size = group_size or int(m.group(2))

    if bits is None:
        m = re.search(r"int(\d+)", quant_path)
        if m:
            bits = int(m.group(1))

    if bits is None:
        bits = 8
    if group_size is None:
        group_size = 128
    return cli_cfg, quant_type, int(bits), int(group_size)


# ---------------- Model loaders (extended to support tp_devices) ----------------
def load_quantized_model(
    path: str,
    device: str = "cuda",
    use_device_map: bool = True,
    cuda_ids: Optional[List[int]] = None,
):
    model, tokenizer = load_tritonv2_quant.load_tritonv2_quantized_model(
        model_dir=path,
        device=device,
        dtype="float16",
        use_device_map=use_device_map,
        cuda_ids=cuda_ids,
    )
    return model, tokenizer


def load_hf_ckpt_qat_model(
    quant_path: str,
    tp_devices: List[int],
    max_mem_ratio: float = 0.85,
):
    """
    QAT HF ckpt: build model on meta, convert Linear->IntQuantLinear, load ckpt with accelerate,
    then dispatch across tp_devices (layer-wise sharding).
    """
    from EfficientQAT.core.quantizer.config import QuantConfig as EQuantConfig

    cli_cfg, quant_type, bits, group_size = _infer_qat_config_from_path(quant_path)
    qcfg = EQuantConfig(
        quant_type=quant_type,
        n_bits=bits,
        group_size=group_size,
        clamp_method=cli_cfg.get("clamp_method", "STE"),
        round_method=cli_cfg.get("round_method", "ste"),
        stat_quant=cli_cfg.get("stat_quant", False),
        iterative_freezing=cli_cfg.get("iterative_freezing", False),
        iterative_freezing_sheduler=cli_cfg.get("iterative_freezing_sheduler", "linear"),
        is_tracking=cli_cfg.get("is_tracking", False),
        freeze_momentum=cli_cfg.get("freeze_momentum", 0.004),
        freeze_threshold=cli_cfg.get("freeze_threshold", 0.0),
        interpolate=cli_cfg.get("interpolate", False),
        lora_rank=cli_cfg.get("lora_rank", 0),
        decay_rate=cli_cfg.get("decay_rate", 0.01),
        shrinking_ratio=cli_cfg.get("shrinking_ratio", 0.5),
        ramp_len=cli_cfg.get("ramp_len", 0),
        ramp_mode=cli_cfg.get("ramp_mode", "linear"),
        ramp_sigmoid_a=cli_cfg.get("ramp_sigmoid_a", 10.0),
    )
    skip_names = {"lm_head"}

    # Build empty template first
    with accelerate.init_empty_weights():
        config = AutoConfig.from_pretrained(
            quant_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        try:
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Convert linears
    _convert_linear_with_skip(model, prefix="", config=qcfg, skip_names=skip_names)

    from EfficientQAT.core.linear.int_quant_linear import set_quant_state
    set_quant_state(model, weight_quant=True)

    model.tie_weights()

    # restrict to tp_devices
    mmap = format_max_memory(tp_devices, max_mem_ratio)
    no_split = getattr(model, "_no_split_modules", None) or []

    device_map = accelerate.infer_auto_device_map(
        model,
        no_split_module_classes=no_split,
        max_memory=mmap,
    )

    # load checkpoint into dispatched model
    try:
        accelerate.load_checkpoint_in_model(
            model,
            checkpoint=quant_path,
            device_map=device_map,
            dtype=torch.float16,
            offload_state_dict=True,
        )
    except TypeError:
        accelerate.load_checkpoint_in_model(
            model,
            checkpoint=quant_path,
            device_map=device_map,
            offload_state_dict=True,
        )

    model = accelerate.dispatch_model(model, device_map=device_map)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(quant_path, local_files_only=True, trust_remote_code=True)
    return model, tokenizer


def load_fp16_hf_model_multi_gpu(
    quant_path: str,
    tp_devices: List[int],
    max_mem_ratio: float = 0.85,
):
    """
    Standard HF fp16 model: rely on transformers device_map="auto" + max_memory to shard across tp_devices.
    """
    mmap = format_max_memory(tp_devices, max_mem_ratio)
    model = AutoModelForCausalLM.from_pretrained(
        quant_path,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=mmap,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def gptq_model_from_path(path, _wbit=2):
    from gptqmodel import GPTQModel
    model = GPTQModel.load(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def qtip_model_from_hf_path(path, tp_devices: List[int], max_mem_ratio: float = 0.85):
    """
    qtip loader: if it supports device_map, shard across tp_devices; else fallback to first device.
    """
    from opponent.qtip.model.llama import LlamaForCausalLM
    mmap = format_max_memory(tp_devices, max_mem_ratio)
    model = LlamaForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory=mmap,
    )
    return model


def load_model_and_tokenizer(quant_path: str, tp_devices: List[int], max_mem_ratio: float) -> Tuple[Any, Any, torch.device]:
    """
    Returns: (model, tokenizer, input_device)
    input_device = where we place input_ids (choose first tp device).
    """
    quant_path = str(quant_path)
    primary = torch.device(f"cuda:{tp_devices[0]}")
    primary_str = str(primary)

    # TritonV2 packed quantized model
    if os.path.isfile(os.path.join(quant_path, "quantize_config.json")):
        use_device_map = len(tp_devices) > 1
        model, tokenizer = load_quantized_model(
            quant_path,
            device=primary_str,
            use_device_map=use_device_map,
            cuda_ids=tp_devices if use_device_map else None,
        )
        return model, tokenizer, primary

    # QAT hf_ckpt
    if "hf_ckpt" in Path(quant_path).parts:
        model, tokenizer = load_hf_ckpt_qat_model(quant_path, tp_devices=tp_devices, max_mem_ratio=max_mem_ratio)
        return model, tokenizer, primary

    # qtip
    if "qtip" in quant_path:
        model = qtip_model_from_hf_path(quant_path, tp_devices=tp_devices, max_mem_ratio=max_mem_ratio)
        tokenizer = AutoTokenizer.from_pretrained(quant_path)
        return model, tokenizer, primary

    # gptq
    if "gptq" in quant_path:
        model, tokenizer = gptq_model_from_path(quant_path)
        # GPTQModel may not support sharding; move to primary
        if hasattr(model, "to"):
            model.to(primary)
        return model, tokenizer, primary

    # default fp16 HF
    print("Loading HF fp16 model from", quant_path)
    if len(tp_devices) == 1:
        model = AutoModelForCausalLM.from_pretrained(
            quant_path,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.to(primary)
        tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
        model.eval()
        return model, tokenizer, primary

    model, tokenizer = load_fp16_hf_model_multi_gpu(quant_path, tp_devices=tp_devices, max_mem_ratio=max_mem_ratio)
    return model, tokenizer, primary


# ---------------- Evaluation ----------------
@torch.no_grad()
def get_logits_for_causal_lm(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Prefer model(...) logits when available; fallback to model.model + lm_head.
    """
    try:
        outputs = model(input_ids, use_cache=False)
    except TypeError:
        outputs = model(input_ids)
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if hasattr(model, "model") and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        outputs = model.model(input_ids, use_cache=False)
        hidden = outputs[0]
        logits = model.lm_head(hidden.to(model.lm_head.weight.dtype))
        return logits
    raise RuntimeError("Cannot extract logits from model outputs.")


@torch.no_grad()
def test_ppl_batched(
    model: Any,
    tokenizer: Any,
    datasets: List[str],
    ppl_seqlen: int,
    batch_size: int,
    input_device: torch.device,
    show_progress: bool = True,
) -> Dict[str, float]:
    results: Dict[str, float] = {}

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    model.eval()

    def _loss_sum_from_model(batch: torch.Tensor):
        try:
            outputs = model(batch, labels=batch, use_cache=False)
        except TypeError:
            outputs = model(batch, labels=batch)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            return None
        tok_cnt = batch.size(0) * (batch.size(1) - 1)
        return loss.float() * tok_cnt, tok_cnt

    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True,
        )
        testenc = testloader if ("c4" in dataset) else testloader.input_ids

        tokens = testenc.reshape(-1)
        seqlen = ppl_seqlen
        nsamples = tokens.numel() // seqlen
        tokens = tokens[: nsamples * seqlen]
        samples = tokens.view(nsamples, seqlen)  # [nsamples, seqlen]

        if nsamples == 0:
            print(f"[WARN] {dataset}: no samples (tokens={tokens.numel()}, seqlen={seqlen})")
            results[dataset] = float("nan")
            continue

        s, e = 0, nsamples

        local_nll_sum = None
        local_tok_cnt = 0

        for i in tqdm(range(s, e, batch_size), desc=f"ppl[{dataset}]", leave=False, disable=not show_progress):
            batch = samples[i : min(e, i + batch_size)].to(input_device, non_blocking=True)  # [bs, seqlen]

            loss_pack = _loss_sum_from_model(batch)
            if loss_pack is None or not torch.isfinite(loss_pack[0]).item():
                logits = get_logits_for_causal_lm(model, batch)  # [bs, seqlen, vocab]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                if shift_labels.device != shift_logits.device:
                    shift_labels = shift_labels.to(shift_logits.device)

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                if not torch.isfinite(loss).item() and shift_logits.dtype != torch.float32:
                    loss = F.cross_entropy(
                        shift_logits.float().view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="sum",
                    )
                loss_sum = loss.double()
                tok_cnt = shift_labels.numel()
            else:
                loss_sum, tok_cnt = loss_pack
                loss_sum = loss_sum.double()

            if local_nll_sum is None:
                local_nll_sum = torch.zeros((), device=loss_sum.device, dtype=torch.float64)
            local_nll_sum += loss_sum
            local_tok_cnt += tok_cnt

        if local_tok_cnt == 0:
            print(f"[WARN] {dataset}: token count is zero after batching")
            results[dataset] = float("nan")
            continue

        avg_nll = float(local_nll_sum) / local_tok_cnt
        if not math.isfinite(avg_nll):
            print(f"[WARN] {dataset}: non-finite avg_nll={avg_nll}")
            results[dataset] = float("inf")
            continue
        try:
            ppl = math.exp(avg_nll)
        except OverflowError:
            ppl = float("inf")
        results[dataset] = ppl

    model.config.use_cache = use_cache
    return results


@torch.no_grad()
def run_lm_eval_tasks(
    model: Any,
    tasks_csv: str,
    num_fewshot: int,
    eval_batch_size: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    lm_eval: keep as-is; for multi-gpu sharded models it may work depending on HFLM wrapper.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    set_eval_seed(seed)
    task_list = [t.strip() for t in tasks_csv.split(",") if t.strip()]
    task_manager = TaskManager()

    model_eval = HFLM(pretrained=model, batch_size=eval_batch_size)

    eval_results = lm_eval.simple_evaluate(
        model=model_eval,
        tasks=task_list,
        num_fewshot=num_fewshot,
        task_manager=task_manager,
        log_samples=False,
    )
    return eval_results


def build_eval_args() -> EasyDict:
    args = EasyDict()
    args["train_param_settings"] = {}
    train_params = args["train_param_settings"]
    train_params["eval_ppl"] = True
    train_params["ppl_seqlen"] = 2048
    train_params["batch_size"] = 8
    train_params["eval_tasks"] = "mmlu"  # "" to disable
    train_params["num_fewshot"] = 5
    args.eval_batch_size = 16
    return args


# ---------------- Your QUANT_PATHS ----------------
QUANT_PATHS = [
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual/checkpoints/out",
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-end025/checkpoints/out",
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-end050/checkpoints/out",
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-end075/checkpoints/out",
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-int2/checkpoints/out",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tp-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    p.add_argument("--max-mem-ratio", type=float, default=float(os.environ.get("MAX_MEM_RATIO", "0.85")))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-eval-tasks", action="store_true")
    return p.parse_args()


def main():
    args_cli = parse_args()
    dist_cfg = init_distributed(args_cli.tp_size)
    is_rank0 = dist_cfg.rank == 0

    pcfg = ParallelConfig(
        tp_size=args_cli.tp_size,
        max_mem_ratio=args_cli.max_mem_ratio,
        compile_model=args_cli.compile,
    )

    if dist_cfg.enabled:
        tp_devices = [dist_cfg.local_rank]
        primary_device_id = dist_cfg.local_rank
    else:
        tp_devices = pick_tp_devices(tp_size=pcfg.tp_size)
        primary_device_id = tp_devices[0]
    tp_summary_devices = list(range(pcfg.tp_size)) if dist_cfg.enabled else tp_devices
    set_primary_device(primary_device_id)
    primary = torch.device(f"cuda:{primary_device_id}")

    # performance knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    rank0_print(is_rank0, f"[Topology] tp_size={pcfg.tp_size}")
    rank0_print(is_rank0, f"[TP] tp_devices={tp_devices}, primary={primary}")

    eval_cfg = build_eval_args()
    if args_cli.no_eval_tasks:
        eval_cfg["train_param_settings"]["eval_tasks"] = ""

    all_results: Dict[str, Any] = {}

    for quant_path in QUANT_PATHS:
        model, tokenizer, input_device = load_model_and_tokenizer(
            quant_path,
            tp_devices=tp_devices,
            max_mem_ratio=pcfg.max_mem_ratio,
        )

        if dist_cfg.enabled:
            from EfficientQAT.core.linear.q_linear_pack import PackableQuantLinear

            if any(isinstance(m, PackableQuantLinear) for m in model.modules()):
                apply_tp_for_eval(model, dist_cfg.tp_mesh)
                rank0_print(is_rank0, f"[TP] apply_tp_for_eval enabled for {quant_path}")

        if pcfg.compile_model:
            # compile can break for some custom modules; keep optional
            try:
                model = torch.compile(model, mode="max-autotune")
                rank0_print(is_rank0, f"torch.compile enabled for {quant_path}")
            except Exception as e:
                rank0_print(is_rank0, f"torch.compile failed for {quant_path}: {e}")

        # PPL
        ppl_results = {}
        if eval_cfg["train_param_settings"]["eval_ppl"]:
            ppl_results = test_ppl_batched(
                model=model,
                tokenizer=tokenizer,
                datasets=["wikitext2", "c4"],
                ppl_seqlen=int(eval_cfg["train_param_settings"].get("ppl_seqlen", 2048)),
                batch_size=int(eval_cfg["train_param_settings"].get("batch_size", 8)),
                input_device=input_device,
                show_progress=is_rank0,
            )

        # tasks
        tasks_csv = eval_cfg["train_param_settings"].get("eval_tasks", "")
        task_results = None
        if tasks_csv:
            try:
                task_results = run_lm_eval_tasks(
                    model=model,
                    tasks_csv=tasks_csv,
                    num_fewshot=int(eval_cfg["train_param_settings"]["num_fewshot"]),
                    eval_batch_size=int(eval_cfg.get("eval_batch_size", 16)),
                    seed=0,
                )
            except Exception as e:
                task_results = {"error": str(e)}

        # record
        all_results[quant_path] = {
            "ppl": ppl_results,
            "tasks": task_results,
            "tp_devices": tp_summary_devices,
        }

        # cleanup
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if dist_cfg.enabled:
        dist.barrier()

    if is_rank0:
        print("\n========== FINAL RESULTS ==========")
        for path, res in all_results.items():
            print(f"\n[MODEL] {path}")
            print(f"  TP_DEVICES: {res.get('tp_devices')}")
            if res.get("ppl"):
                for ds, v in res["ppl"].items():
                    print(f"  PPL {ds}: {v:.4f}")
            if res.get("tasks") is not None:
                # keep it compact; you can dump make_table outside if needed
                if isinstance(res["tasks"], dict) and "error" in res["tasks"]:
                    print(f"  TASKS ERROR: {res['tasks']['error']}")
                else:
                    print("  TASKS: done (see raw dict if you want to print tables)")

"""
torchrun --nproc_per_node=4 test/eval_batch.py --tp-size 4
"""
if __name__ == "__main__":
    main()
