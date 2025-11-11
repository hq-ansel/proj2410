"""
Sampling primitives shared across EfficientQAT.

This module contains the generic components previously embedded inside
`data_sampler.py`. By centralising them we reduce duplication and give other
modules (dataset builders, unit tests, CLI tools) a single import surface for
chunking and sampling text streams.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Iterator, List, Optional, Protocol, Type, TypeVar



__all__ = [
    "TextSource",
    "Tokenizer",
    "Writer",
    "ListTextSource",
    "FilePerLineTextSource",
    "JSONLWriter",
    "Chunker",
    "ReservoirSampler",
    "SamplerConfig",
    "SamplerPipeline",
    "extract_random_chunks",
    "MockTokenizer",
]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------

class ListTextSource:
    """TextSource: from a python list of strings."""

    def __init__(self, texts: Iterable[str]) -> None:
        self._texts = texts

    def __iter__(self) -> Iterator[str]:
        for text in self._texts:
            yield text


class FilePerLineTextSource:
    """TextSource: each line in a text file is a unit."""

    def __init__(self, path: Path, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding

    def __iter__(self) -> Iterator[str]:
        with self.path.open("r", encoding=self.encoding) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped


class JSONLWriter:
    """Writer: emits JSONL with decoded text (and optional token ids)."""

    def __init__(
        self,
        path: Path,
        include_ids: bool = False,
        encoding: str = "utf-8",
    ) -> None:
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


# ---------------------------------------------------------------------------
# Core sampling components
# ---------------------------------------------------------------------------


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
            chunk = self._buffer[: self.seq_len]
            self._buffer = self._buffer[self.seq_len :]
            yield chunk

    def _yield_overlap(self) -> Iterator[List[int]]:
        # When stride>0, window step = seq_len - stride
        step = self.seq_len - self.stride
        while len(self._buffer) >= self.seq_len:
            chunk = self._buffer[: self.seq_len]
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


# ---------------------------------------------------------------------------
# High-level pipeline abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerConfig:
    seq_len: int = 4096
    num_samples: int = 1024
    stride: int = 0
    seed: Optional[int] = 42

    def validate(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if self.stride < 0 or self.stride >= self.seq_len:
            raise ValueError("stride must satisfy 0 <= stride < seq_len")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0")


ChunkerT = TypeVar("ChunkerT", bound=Chunker)
SamplerT = TypeVar("SamplerT", bound=ReservoirSampler)


class SamplerPipeline:
    """
    Declarative wrapper around Chunker + ReservoirSampler.

    This class knows how to:
      - Build the chunk/sampler objects from configuration
      - Run the sampling loop
      - Optionally persist results via a Writer
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: SamplerConfig,
        *,
        chunker_cls: Type[ChunkerT] = Chunker,
        sampler_cls: Type[SamplerT] = ReservoirSampler,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.config.validate()
        self.chunker_cls = chunker_cls
        self.sampler_cls = sampler_cls

    def build_chunker(self) -> ChunkerT:
        return self.chunker_cls(
            tokenizer=self.tokenizer,
            seq_len=self.config.seq_len,
            stride=self.config.stride,
        )

    def build_sampler(self) -> SamplerT:
        return self.sampler_cls(k=self.config.num_samples, seed=self.config.seed)

    def iter_chunks(self, text_source: TextSource) -> Iterator[List[int]]:
        chunker = self.build_chunker()
        return chunker.chunk_stream(iter(text_source))

    def sample(self, text_source: TextSource) -> List[List[int]]:
        sampler = self.build_sampler()
        return sampler.sample(self.iter_chunks(text_source))

    def run(
        self,
        text_source: TextSource,
        writer: Optional[Writer] = None,
    ) -> List[List[int]]:
        chunks = self.sample(text_source)
        if writer is not None:
            for ids in chunks:
                writer.write(ids, self.tokenizer)
        return chunks


# ---------------------------------------------------------------------------
# Mock tokenizer (useful for tests / CLI defaults)
# ---------------------------------------------------------------------------


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
        for tok in toks:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._rev)
                self._rev.append(tok)
            ids.append(self._vocab[tok])
        return ids

    def decode(self, ids: List[int]) -> str:
        parts = []
        for idx in ids:
            if 0 <= idx < len(self._rev):
                parts.append(self._rev[idx])
            else:
                parts.append("<UNK>")
        return " ".join(parts)
