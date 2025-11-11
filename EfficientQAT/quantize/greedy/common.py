from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..utils import Catcher

__all__ = [
    "CosineAnnealingScheduler",
    "CatcherManager",
    "CommonInputDataset",
]


class CosineAnnealingScheduler:
    """
    Lightweight cosine scheduler that supports both decay (max -> min) and
    warmup (0 -> max) behaviours. Greedy training uses it for dampening loss
    weighting during early iterations.
    """

    def __init__(
        self,
        max_value: float,
        min_value: float = 0.0,
        total_steps: int = 100,
        *,
        ascend: bool = False,
    ) -> None:
        self.max_value = max_value
        self.min_value = min_value
        self.total_steps = max(1, total_steps)
        self.current_step = 0
        self.ascend = ascend

    def step(self) -> float:
        step = min(self.current_step, self.total_steps)
        if self.ascend:
            value = self.max_value * 0.5 * (1 - math.cos(math.pi * step / self.total_steps))
        else:
            value = self.min_value + 0.5 * (self.max_value - self.min_value) * (
                1 + math.cos(math.pi * step / self.total_steps)
            )
        self.current_step += 1
        return value


class CatcherManager:
    """
    Context manager that temporarily swaps target layers with Catcher modules so
    we can record intermediate activations without permanently mutating the
    model. Used by both lazy datasets and with-catcher training flows.
    """

    def __init__(self, layers: nn.ModuleList, indices: Iterable[int]) -> None:
        self.layers = layers
        self.indices = list(indices)
        self.original_modules: Dict[int, nn.Module] = {}

    def __enter__(self):
        for idx in self.indices:
            if isinstance(self.layers[idx], Catcher):
                continue
            self.original_modules[idx] = self.layers[idx]
            self.layers[idx] = Catcher(self.layers[idx])

    def __exit__(self, exc_type, exc_val, exc_tb):
        for idx in self.indices:
            if isinstance(self.layers[idx], Catcher):
                self.layers[idx] = self.original_modules[idx]


class CommonInputDataset(Dataset):
    """Wraps a list of tensors so they can be fed through DataLoader."""

    def __init__(self, data: List[torch.Tensor]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        res = self.data[idx]
        if len(res.shape) == 2:
            res = res.squeeze(0)
        return res
