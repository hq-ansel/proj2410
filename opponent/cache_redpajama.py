#!/usr/bin/env python3
"""
Builds a cached calibration set from a local RedPajama Arrow shard.

Default behavior mirrors the manual steps we used during planning:
  - load tokenizer from /home/ubuntu/data/exp/proj2410/model/Llama2-7b
  - read arrow file red_pajama-data-1_t-sample-train-00000-of-00337.arrow
  - sample 1024 sequences of length 4096 with the same slicing logic as
    opponent/AQLM/src/datautils.py:get_red_pajama
  - save to /home/ubuntu/data/exp/proj2410/cache/redpajama_stream/manual_aqlm_1024_4096.pt
"""

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer


DEFAULT_MODEL = "/home/ubuntu/data/exp/proj2410/model/Llama2-7b"
DEFAULT_ARROW = (
    "/home/ubuntu/data/exp/proj2410/"
    "hf_home/datasets/red_pajama-data-1_t-sample/default/0.0.0/"
    "05f7d4c498dee422/red_pajama-data-1_t-sample-train-00000-of-00337.arrow"
)
DEFAULT_OUTPUT = (
    "/home/ubuntu/data/exp/proj2410/cache/redpajama_stream/manual_aqlm_1024_4096.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache RedPajama slices locally.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Tokenizer source (default: %(default)s)",
    )
    parser.add_argument(
        "--arrow",
        default=DEFAULT_ARROW,
        help="Path to a single Arrow shard (default: %(default)s)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="Number of sequences to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=4096,
        help="Sequence length to slice (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output .pt file for cached tensors (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"[cache] loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.bos_token_id = tokenizer.bos_token_id or 1
    tokenizer.eos_token_id = tokenizer.eos_token_id or 2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[cache] using CUDA acceleration on {gpu_name}")
    else:
        print("[cache] CUDA unavailable; defaulting to CPU sampling")

    print(f"[cache] loading Arrow shard {args.arrow}")
    dataset = Dataset.from_file(args.arrow)
    if len(dataset) == 0:
        raise RuntimeError(f"No records found in shard {args.arrow}")

    samples = []
    progress = 0
    print(
        f"[cache] sampling {args.nsamples} sequences of length {args.seqlen} "
        f"(seed={args.seed})"
    )
    while len(samples) < args.nsamples:
        idx = rng.randint(0, len(dataset) - 1)
        text = dataset[int(idx)].get("text") or ""
        if not text:
            continue

        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] <= args.seqlen + 1:
            continue

        max_start = input_ids.shape[1] - args.seqlen - 1
        start = rng.randint(0, max_start)
        end = start + args.seqlen
        samples.append(input_ids[:, start:end].contiguous().cpu())

        if len(samples) // 50 > progress:
            progress = len(samples) // 50
            print(f"[cache] collected {len(samples)} samples")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output_path)
    print(f"[cache] saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
