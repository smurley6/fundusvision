"""Device selection and seeding helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU.

    ``prefer`` forces a specific device string (e.g. ``"cpu"``) when given.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
