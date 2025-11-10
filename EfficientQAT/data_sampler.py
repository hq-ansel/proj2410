#!/usr/bin/env python3
# rpj_sampler.py
# Single-file, SRP-friendly, tokenizer-agnostic 4096-token chunk sampler.

from __future__ import annotations
from typing import Iterable, Iterator, List, Protocol, Optional
import random
import json
import argparse
from pathlib import Path

# =========================
# 1) Abstract Protocols
# =========================

class TextSource(Protocol):
    """Produces a stream of raw text units (docs/lines/paragraphs)."""
    def __iter__(self) -> Iterator[str]: ...


class Tokenizer(Protocol):
    """Tokenizer-agnostic interface."""
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...


class Writer(Protocol):
    """Persist sampled chunks."""
    def write(self, chunk_token_ids: List[int], tokenizer: Tokenizer) -> None: ...
    def close(self) -> None: ...


# =========================
# 2) Concrete, Minimal Adapters (optional)
# =========================

class ListTextSource:
    """TextSource: from a python list of strings."""
    def __init__(self, texts: Iterable[str]) -> None:
        self._texts = texts
    def __iter__(self) -> Iterator[str]:
        for t in self._texts:
            yield t


class FilePerLineTextSource:
    """TextSource: each line in a text file is a unit."""
    def __init__(self, path: Path, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding
    def __iter__(self) -> Iterator[str]:
        with self.path.open("r", encoding=self.encoding) as f:
            for line in f:
                if line.strip():
                    yield line.rstrip("\n")


class JSONLWriter:
    """Writer: emits JSONL with decoded text (and optional token ids)."""
    def __init__(self, path: Path, include_ids: bool = False, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding
        self.include_ids = include_ids
        self._fh = self.path.open("w", encoding=self.encoding)
    def write(self, chunk_token_ids: List[int], tokenizer: Tokenizer) -> None:
        obj = {"text": tokenizer.decode(chunk_token_ids)}
        if self.include_ids:
            obj["ids"] = chunk_token_ids
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    def close(self) -> None:
        self._fh.close()


# =========================
# 3) Core SRP Components
# =========================

class Chunker:
    """
    Turns a stream of texts into fixed-length token chunks of `seq_len`.
    - No padding. Remainder is kept in buffer.
    - Optional overlap via `stride` (< seq_len).
    """
    def __init__(self, tokenizer: Tokenizer, seq_len: int, stride: int = 0) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if stride < 0 or stride >= seq_len:
            raise ValueError("stride must satisfy 0 <= stride < seq_len")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self._buffer: List[int] = []

    def _yield_nonoverlap(self) -> Iterator[List[int]]:
        while len(self._buffer) >= self.seq_len:
            chunk = self._buffer[:self.seq_len]
            self._buffer = self._buffer[self.seq_len:]
            yield chunk

    def _yield_overlap(self) -> Iterator[List[int]]:
        # When stride>0, window step = seq_len - stride
        step = self.seq_len - self.stride
        while len(self._buffer) >= self.seq_len:
            chunk = self._buffer[:self.seq_len]
            self._buffer = self._buffer[step:]
            yield chunk

    def chunk_stream(self, text_stream: Iterable[str]) -> Iterator[List[int]]:
        for text in text_stream:
            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            self._buffer.extend(ids)
            if self.stride == 0:
                yield from self._yield_nonoverlap()
            else:
                yield from self._yield_overlap()
        # By design: leftover tokens are discarded (no pad).


class ReservoirSampler:
    """
    Reservoir sampling over an unknown-length chunk stream.
    Guarantees uniform selection of k items.
    """
    def __init__(self, k: int, seed: Optional[int] = 42) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.seed = seed

    def sample(self, stream: Iterable[List[int]]) -> List[List[int]]:
        if self.seed is not None:
            random.seed(self.seed)
        reservoir: List[List[int]] = []
        seen = 0
        for chunk in stream:
            seen += 1
            if len(reservoir) < self.k:
                reservoir.append(chunk)
            else:
                j = random.randint(1, seen)
                if j <= self.k:
                    reservoir[j - 1] = chunk
        return reservoir


# =========================
# 4) Orchestrator (pipeline)
# =========================

def extract_random_chunks(
    text_source: TextSource,
    tokenizer: Tokenizer,
    seq_len: int,
    num_samples: int,
    stride: int = 0,
    seed: Optional[int] = 42,
) -> List[List[int]]:
    """
    End-to-end pipeline:
      TextSource -> Chunker -> ReservoirSampler -> [num_samples x seq_len]
    """
    chunker = Chunker(tokenizer, seq_len=seq_len, stride=stride)
    sampler = ReservoirSampler(k=num_samples, seed=seed)
    return sampler.sample(chunker.chunk_stream(iter(text_source)))


# =========================
# 5) Demo Tokenizer (for quick testing only)
# =========================

class MockTokenizer:
    """
    Example tokenizer (NOT for production):
      - Splits by whitespace into "tokens"
      - Maps each distinct token to an integer id deterministically
    Replace with your real tokenizer (HF/tiktoken/SentencePiece/…)
    """
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._rev: List[str] = []

    def encode(self, text: str) -> List[int]:
        toks = text.strip().split()
        ids: List[int] = []
        for t in toks:
            if t not in self._vocab:
                self._vocab[t] = len(self._rev)
                self._rev.append(t)
            ids.append(self._vocab[t])
        return ids

    def decode(self, ids: List[int]) -> str:
        parts = []
        for i in ids:
            if 0 <= i < len(self._rev):
                parts.append(self._rev[i])
            else:
                parts.append("<UNK>")
        return " ".join(parts)


# =========================
# 6) CLI (optional)
# =========================

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tokenizer-agnostic sampler: sample N chunks of length L from a text stream."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input text file (one document/line).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL file.")
    p.add_argument("--seq-len", type=int, default=4096,
                   help="Fixed chunk token length (default: 4096).")
    p.add_argument("--num-samples", type=int, default=1024,
                   help="Number of chunks to sample (default: 1024).")
    p.add_argument("--stride", type=int, default=0,
                   help="Overlap tokens (0 = no overlap). Must be < seq_len.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reservoir sampling.")
    p.add_argument("--include-ids", action="store_true",
                   help="Include raw token ids in JSONL.")
    return p

def main() -> None:
    args = _build_cli().parse_args()

    # Replace MockTokenizer with your real tokenizer implementation:
    tokenizer: Tokenizer = MockTokenizer()

    source: TextSource = FilePerLineTextSource(args.input)
    chunks = extract_random_chunks(
        text_source=source,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        stride=args.stride,
        seed=args.seed,
    )

    writer: Writer = JSONLWriter(args.output, include_ids=args.include_ids)
    try:
        for ids in chunks:
            writer.write(ids, tokenizer)
    finally:
        writer.close()

if __name__ == "__main__":
    main()
