"""Shared training utilities: JSON I/O, checkpoints and early stopping."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_weights(model: nn.Module, path: str | Path) -> None:
    """Save a pure ``state_dict`` (loads under ``torch.load(weights_only=True)``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(
    model: nn.Module, path: str | Path, device: torch.device | str = "cpu"
) -> nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


class EarlyStopping:
    """Track the best value of a metric and stop after ``patience`` epochs without improvement."""

    def __init__(self, patience: int, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best: float | None = None
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Record ``value`` for ``epoch``; return ``True`` if it is a new best."""
        improved = self.best is None or (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience
