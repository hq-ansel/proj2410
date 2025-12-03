#!/usr/bin/env python3
"""
Reusable helpers + experiment plan for comparing AQLM, GPTQModel, QTIP, and VPTQ
on Llama2-7B with a shared streaming RedPajama calibration set.

改造点：
1. 采样时只保存“文本样本”（HF Datasets Arrow），与 tokenizer 无关；
2. 不同 tokenizer 通过同一份文本缓存生成各自的 token cache（.pt）；
3. 上层 plan_* 接口保持不变：依然拿到的是一个 token cache 路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import trange
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "Llama2-7b"
QUANT_ROOT = PROJECT_ROOT / "quant_model" / "Llama2-7B"

# token cache（按 tokenizer 区分）保存位置
CACHE_ROOT = PROJECT_ROOT / "cache" / "redpajama_stream"
# 文本缓存（只采样一次，共享）保存位置
HF_HOME = PROJECT_ROOT / "hf_home"

DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
DATASET_SPLIT = "train"
SHUFFLE_BUFFER = 10_000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmPlan:
    name: str
    dataset_cache: Path
    output_dir: Path
    command: str
    notes: List[str]

    def as_dict(self) -> Dict[str, str]:
        payload = asdict(self)
        payload["dataset_cache"] = str(self.dataset_cache)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_tokenizer(tokenizer_id: str) -> str:
    return tokenizer_id.strip("/").replace("/", "__")


# ---------------------------------------------------------------------------
# Text cache（与 tokenizer 无关，只采样一次）
# ---------------------------------------------------------------------------

def _cache_name_text(
    nsamples: int,
    seqlen: int,
    seed: int,
    dataset_name: str,
    split: str,
) -> Path:
    """
    文本缓存路径（HF Datasets save_to_disk 目录，不包含 tokenizer_id）。
    """
    base = _ensure_dir(HF_HOME / "rp_text_caches")
    # 简单一点就直接拼在一起，如果 worried about 路径过长可以再 hash 一层
    fname = f"redpajama_text_ns{nsamples}_sl{seqlen}_seed{seed}_{split}"
    # 目录名随便，扩展名不重要；这里不加 .arrow 更直观
    return base / fname


def _materialize_text_cache(
    cache_path: Path,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
    dataset_name: str,
    split: str,
) -> None:
    """
    从 RedPajama 抽样 nsamples 条文本，保存成 HF Dataset（Arrow）。

    流程：
      1) 优先尝试 HF streaming。
      2) 如果 streaming 被 block（比如本地脚本 / 离线环境），
         回退到本地 hf_home/datasets 下的 RedPajama 数据目录：
         - 优先用 load_from_disk 直接加载整个数据集目录；
         - map-style 模式下，用「打乱索引 + 顺序访问」抽样。
    """
    print(f"[stream-text] building text cache {cache_path} ({nsamples} samples)")
    rng = random.Random(seed)

    dataset = None
    iterator = None
    is_iterable = True

    # ------------------------------------------------------------------
    # 1. 尝试正常的 HF streaming 方式
    # ------------------------------------------------------------------
    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
        ).shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
        iterator = iter(dataset)
        print("[stream-text] using HF streaming backend")
    except Exception as err:
        # 如果不是默认的 RedPajama，就直接把异常抛出去
        if dataset_name != DATASET_NAME:
            raise

        # ------------------------------------------------------------------
        # 2. 回退到本地 Arrow / HF 缓存目录
        # ------------------------------------------------------------------
        arrow_dir = _local_redpajama_dir()
        if not arrow_dir:
            raise RuntimeError(
                "Failed to stream RedPajama and no local Arrow cache found under hf_home."
            ) from err

        print(
            "[stream-text] remote streaming blocked, falling back to local Arrow data dir:\n"
            f"  directory: {arrow_dir}"
        )

        # 🔴 关键：不要手动挑某个 .arrow 文件，而是按 HF 规范加载整个数据集目录
        try:
            from datasets import load_from_disk

            dataset = load_from_disk(str(arrow_dir))
            is_iterable = False
            dataset_size = len(dataset)
        except Exception as err2:
            # 如果 load_from_disk 也挂了，再退一步用单 shard .arrow
            from datasets import Dataset as HFDataset

            arrow_files = sorted(
                arrow_dir.glob("red_pajama-data-1_t-sample-train-*.arrow")
            )
            if not arrow_files:
                # 兜底：任何 .arrow 都没找到就报错
                any_arrow = sorted(arrow_dir.glob("*.arrow"))
                if not any_arrow:
                    raise RuntimeError(
                        f"No Arrow shards found under {arrow_dir}"
                    ) from err2
                arrow_files = any_arrow

            local_arrow = arrow_files[0]
            print(
                "[stream-text] load_from_disk failed; falling back to single shard:\n"
                f"  shard: {local_arrow.name}"
            )
            dataset = HFDataset.from_file(str(local_arrow))
            is_iterable = False
            dataset_size = len(dataset)

        if dataset is None:
            raise RuntimeError(
                f"Failed to construct local RedPajama dataset under {arrow_dir}"
            )

        if dataset_size == 0:
            raise RuntimeError(f"Local dataset at {arrow_dir} is empty") from err

        print(f"[stream-text] local map-style dataset size = {dataset_size}")

    # ------------------------------------------------------------------
    # 3. 抽样逻辑：iterable / map-style 两种后端
    # ------------------------------------------------------------------
    texts = []
    log_every = max(1, min(64, nsamples // 8 or 1))  # 动态决定一下多少条打印一次日志

    # map-style 数据集：预先生成一个打乱的索引序列，再顺序访问
    if not is_iterable:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        rng.shuffle(indices)
        idx_ptr = 0

    while len(texts) < nsamples:
        if is_iterable:
            # streaming iterable dataset
            try:
                row = next(iterator)
            except StopIteration:
                # streaming 的话就重新 shuffle 一遍
                dataset = dataset.shuffle(
                    seed=rng.randint(0, 2**31 - 1),
                    buffer_size=SHUFFLE_BUFFER,
                )
                iterator = iter(dataset)
                continue
        else:
            # map-style Arrow 数据：打乱后顺序访问
            if idx_ptr >= len(indices):
                rng.shuffle(indices)
                idx_ptr = 0
            row = dataset[int(indices[idx_ptr])]
            idx_ptr += 1

        text = row.get("text") or row.get("content") or ""
        if not text or not text.strip():
            continue

        # 简单裁一下长度，避免极端长文本
        texts.append({"text": text})

        if len(texts) % log_every == 0:
            print(
                f"[stream-text] collected {len(texts)}/{nsamples} samples...",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 4. 保存到磁盘：真正写入 cache_path
    # ------------------------------------------------------------------
    ds = Dataset.from_list(texts)
    ds.save_to_disk(str(cache_path))
    print(f"[stream-text] cached {len(texts)} raw samples to {cache_path}")




def ensure_text_cache(
    nsamples: int,
    seqlen: int,
    *,
    seed: int = 42,
    dataset_name: str = DATASET_NAME,
    split: str = DATASET_SPLIT,
    build: bool = False,
) -> Path:
    """
    确保文本缓存存在：采样结果与 tokenizer 无关。
    """
    cache_path = _cache_name_text(nsamples, seqlen, seed, dataset_name, split)
    if build and not cache_path.exists():
        _materialize_text_cache(
            cache_path,
            nsamples=nsamples,
            seqlen=seqlen,
            seed=seed,
            dataset_name=dataset_name,
            split=split,
        )
    return cache_path


# ---------------------------------------------------------------------------
# Token cache（依赖 tokenizer，但可复用同一 text cache）
# ---------------------------------------------------------------------------

def _cache_name(
    nsamples: int,
    seqlen: int,
    seed: int,
    tokenizer_id: str,
    dataset_name: str,
    split: str,
) -> Path:
    """
    token cache 的 .pt 路径，包含 tokenizer_id。
    """
    base = _ensure_dir(CACHE_ROOT)
    tok = _sanitize_tokenizer(tokenizer_id)
    # dataset_name 通常比较长，这里 hash 一下避免文件名过长
    ds_hash = hashlib.sha1(dataset_name.encode("utf-8")).hexdigest()[:8]
    fname = f"rp_tok_ns{nsamples}_sl{seqlen}_seed{seed}_{tok}_{split}_{ds_hash}.pt"
    return base / fname


def build_token_cache_from_text_cache(
    text_cache_path: Path,
    token_cache_path: Path,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
    tokenizer_id: str,
) -> None:
    """
    给某个 tokenizer：从共享 text cache 生成 token cache（.pt）。
    """
    print(
        f"[stream] building token cache {token_cache_path} "
        f"from text cache {text_cache_path}"
    )
    ds = load_from_disk(str(text_cache_path))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.bos_token_id = tokenizer.bos_token_id or 1
    tokenizer.eos_token_id = tokenizer.eos_token_id or 2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        print(f"[stream] using CUDA acceleration on {torch.cuda.get_device_name(device)}")
    else:
        print("[stream] CUDA unavailable; tokenizing on CPU")

    rng = random.Random(seed)

    samples: List[torch.Tensor] = []
    pbar = trange(nsamples, desc="Tokenizing text cache", leave=False)

    i = 0
    n_texts = len(ds)
    if n_texts == 0:
        raise RuntimeError(f"Text cache at {text_cache_path} is empty.")

    while len(samples) < nsamples:
        row = ds[i % n_texts]
        i += 1

        text = row["text"]
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        if input_ids.shape[1] <= seqlen + 1:
            continue

        max_start = input_ids.shape[1] - seqlen - 1
        if max_start <= 0:
            continue

        start = rng.randint(0, max_start)
        end = start + seqlen
        chunk = input_ids[:, start:end].contiguous().cpu()
        samples.append(chunk)
        pbar.update(1)

    pbar.close()
    torch.save(samples, token_cache_path)
    print(f"[stream] cached {len(samples)} samples to {token_cache_path}")


def ensure_stream_cache(
    nsamples: int,
    seqlen: int,
    *,
    seed: int = 42,
    tokenizer_id: Optional[str] = None,
    build: bool = False,
    dataset_name: str = DATASET_NAME,
    split: str = DATASET_SPLIT,
) -> Path:
    """
    对上层来说：返回一个 .pt 路径，里面是 tokenized 的 calibration tensors。

    内部逻辑：
    1. 确保 shared text cache 存在（与 tokenizer 无关，只采样一次）；
    2. 如果需要且 token cache 不存在，则基于 text cache + tokenizer 生成。
    """
    tok_id = tokenizer_id or str(MODEL_PATH)
    token_cache_path = _cache_name(nsamples, seqlen, seed, tok_id, dataset_name, split)
    text_cache_path = ensure_text_cache(
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
        dataset_name=dataset_name,
        split=split,
        build=build,
    )

    if build and not token_cache_path.exists():
        build_token_cache_from_text_cache(
            text_cache_path,
            token_cache_path,
            nsamples=nsamples,
            seqlen=seqlen,
            seed=seed,
            tokenizer_id=tok_id,
        )

    return token_cache_path


# 如果你完全不想存 .pt，只保留文本，可以用这个函数现场 token：
def load_cached_samples_text_based(
    text_cache_path: Path,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
    tokenizer_id: str,
):
    ds = load_from_disk(str(text_cache_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    rng = random.Random(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    samples = []
    i = 0
    n_texts = len(ds)
    if n_texts == 0:
        raise RuntimeError(f"Text cache at {text_cache_path} is empty.")

    while len(samples) < nsamples:
        row = ds[i % n_texts]
        i += 1
        text = row["text"]

        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] <= seqlen + 1:
            continue
        max_start = input_ids.shape[1] - seqlen - 1
        if max_start <= 0:
            continue
        start = rng.randint(0, max_start)
        end = start + seqlen
        chunk = input_ids[:, start:end].contiguous()
        samples.append(chunk)

    return samples


# ---------------------------------------------------------------------------
# Legacy local helper (可选，暂时没用到，保留也无妨)
# ---------------------------------------------------------------------------

def _local_redpajama_dir() -> Optional[Path]:
    """
    Returns a locally cached RedPajama directory (Arrow shards) if available.
    Expected layout (already present in repo):
    hf_home/datasets/red_pajama-data-1_t-sample/default/0.0.0/<hash>/
    """
    root = HF_HOME / "datasets" / "red_pajama-data-1_t-sample" / "default" / "0.0.0"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/dataset_info.json"))
    if not candidates:
        return None
    return candidates[-1].parent


# ---------------------------------------------------------------------------
# Downstream consumers (GPTQModel / QTIP / VPTQ)
# ---------------------------------------------------------------------------

def load_cached_samples(cache_path: Path):
    return torch.load(cache_path)


def samples_for_gptqmodel(cache_path: Path):
    dataset = []
    for tensor in load_cached_samples(cache_path):
        seq = tensor.squeeze(0)
        attn = torch.ones_like(seq)
        dataset.append({"input_ids": seq.clone(), "attention_mask": attn})
    return dataset


def samples_for_qtip(cache_path: Path):
    samples = load_cached_samples(cache_path)
    stacked = torch.vstack([sample.squeeze(0) for sample in samples])
    return stacked


def _format_command(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

def plan_aqlm(materialize_cache: bool) -> AlgorithmPlan:
    cache = ensure_stream_cache(
        nsamples=1024,
        seqlen=4096,
        seed=42,
        build=materialize_cache,
    )
    output = _ensure_dir(QUANT_ROOT / "aqlm" / "w2g128")
    cmd = [
        "python",
        str(PROJECT_ROOT / "opponent" / "AQLM" / "main.py"),
        str(MODEL_PATH),
        str(cache),
        "--nsamples=1024",
        "--val_size=128",
        "--num_codebooks=1",
        "--nbits_per_codebook=16",
        "--in_group_size=8",
        "--relative_mse_tolerance=0.01",
        "--finetune_batch_size=32",
        "--finetune_max_epochs=10",
        "--finetune_early_stop=3",
        "--finetune_keep_best",
        "--local_batch_size=1",
        "--offload_activations",
        "--save",
        str(output),
    ]
    notes = [
        "Matches the CLI described in opponent/AQLM/README.md:200-245.",
        "Calibration cache now built via shared text cache + tokenizer-specific token cache.",
        "Set CUDA_VISIBLE_DEVICES before running if multiple GPUs are available.",
    ]
    return AlgorithmPlan("AQLM", cache, output, _format_command(cmd), notes)


def plan_gptqmodel(materialize_cache: bool) -> AlgorithmPlan:
    cache = ensure_stream_cache(
        nsamples=1024,
        seqlen=4096,
        seed=42,
        build=materialize_cache,
    )
    output = _ensure_dir(QUANT_ROOT / "gptq" / "w2g128")
    script = (
        "from pathlib import Path;"
        "from opponent.plan_pipeline import samples_for_gptqmodel, MODEL_PATH;"
        "from gptqmodel import GPTQModel, QuantizeConfig;"
        f"ds = samples_for_gptqmodel(Path('{cache}'));"
        "model = GPTQModel.load(str(MODEL_PATH), QuantizeConfig(bits=2, group_size=128));"
        f"model.quantize(ds, batch_size=1); model.save('{output}')"
    )
    cmd = ["python", "-c", script]
    notes = [
        "Quantization API reference: opponent/GPTQModel/README.md:268-291.",
        "Calibration samples loaded from shared streaming cache (token cache built from text cache).",
    ]
    return AlgorithmPlan("GPTQModel", cache, output, _format_command(cmd), notes)


def plan_qtip(materialize_cache: bool) -> AlgorithmPlan:
    cache = ensure_stream_cache(
        nsamples=384,
        seqlen=4096,
        seed=42,
        build=materialize_cache,
    )
    output = _ensure_dir(QUANT_ROOT / "qtip" / "w2g128")
    cmd = [
        "python",
        str(PROJECT_ROOT / "opponent" / "qtip" / "quantize_llama" / "quantize_finetune_llama.py"),
        "--base_model",
        str(MODEL_PATH),
        "--save_path",
        str(output),
        "--devset_size",
        "384",
        "--ctx_size",
        "4096",
        "--batch_size",
        "16",
        "--L",
        "16",
        "--K",
        "2",
        "--V",
        "2",
        "--decode_mode",
        "quantlut_sym",
        "--sample_proc",
        "4",
    ]
    notes = [
        "sample_rp1t / sample_rp1t_concat in opponent/qtip/lib/utils/data_utils.py "
        "should be wired to use the shared cache if you want identical samples.",
        "This plan keeps the tensor format identical; only the sampling backend changed.",
    ]
    return AlgorithmPlan("QTIP", cache, output, _format_command(cmd), notes)


def plan_vptq(materialize_cache: bool) -> AlgorithmPlan:
    cache = ensure_stream_cache(
        nsamples=512,
        seqlen=4096,
        seed=46,
        build=materialize_cache,
    )
    output = _ensure_dir(QUANT_ROOT / "vptq" / "w2g128")
    cmd = [
        "python",
        "-m",
        "vptq",
        "--model",
        str(output),
        "--prompt",
        "Explain: Do Not Go Gentle into That Good Night",
    ]
    notes = [
        "Follow installation guidance in opponent/VPTQ/README.md:70-122.",
        "Any quantization script borrowed from the algorithm branch must "
        "pull calibration samples via the shared streaming cache.",
    ]
    return AlgorithmPlan("VPTQ", cache, output, _format_command(cmd), notes)


def plan_all(materialize_cache: bool) -> List[AlgorithmPlan]:
    return [
        plan_aqlm(materialize_cache),
        plan_gptqmodel(materialize_cache),
        plan_qtip(materialize_cache),
        plan_vptq(materialize_cache),
    ]


def print_plan(plans: List[AlgorithmPlan]) -> None:
    for item in plans:
        print(f"\n[{item.name}]")
        print(f"cache : {item.dataset_cache}")
        print(f"output: {item.output_dir}")
        print("command:")
        print(f"  {item.command}")
        if item.notes:
            print("notes:")
            for note in item.notes:
                print(f"  - {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization plan helper")
    parser.add_argument(
        "--materialize-cache",
        action="store_true",
        help="Stream and save every required RedPajama cache upfront.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the plan as JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    plans = plan_all(materialize_cache=args.materialize_cache)
    if args.json:
        payload = [plan.as_dict() for plan in plans]
        print(json.dumps(payload, indent=2))
    else:
        print_plan(plans)


if __name__ == "__main__":
    main()
