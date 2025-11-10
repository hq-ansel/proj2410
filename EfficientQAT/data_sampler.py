#!/usr/bin/env python3
"""Tokenizer-agnostic CLI for sampling fixed-length token chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from EfficientQAT.core import (
    FilePerLineTextSource,
    JSONLWriter,
    MockTokenizer,
    SamplerConfig,
    SamplerPipeline,
    TextSource,
    Tokenizer,
    Writer,
)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenizer-agnostic sampler: sample N chunks of length L from a text stream."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input text file (one document/line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Fixed chunk token length (default: 4096).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Number of chunks to sample (default: 1024).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Overlap tokens (0 = no overlap). Must be < seq_len.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reservoir sampling.",
    )
    parser.add_argument(
        "--include-ids",
        action="store_true",
        help="Include raw token ids in JSONL output.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()

    tokenizer: Tokenizer = MockTokenizer()  # Swap with HF/tiktoken tokenizer as needed
    config = SamplerConfig(
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        stride=args.stride,
        seed=args.seed,
    )
    pipeline = SamplerPipeline(tokenizer=tokenizer, config=config)

    source: TextSource = FilePerLineTextSource(args.input)
    writer: Writer = JSONLWriter(args.output, include_ids=args.include_ids)
    try:
        pipeline.run(source, writer=writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
