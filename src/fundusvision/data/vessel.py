"""Retinal vessel segmentation dataset (fundus image + binary vessel mask pairs).

Expected layout (the Kaggle "Fundus Image Dataset for Vessel Segmentation")::

    <root>/
      train/Original/*.png        train/Ground truth/*.png
      test/Original/*.png         test/Ground truth/*.png

Images and masks share filenames. Masks are loaded as grayscale, resized with nearest
neighbour interpolation and binarised at ``> 127`` (1 = vessel, 0 = background).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from fundusvision.data.odir import IMAGENET_MEAN, IMAGENET_STD

IMAGE_SUBDIR = "Original"
MASK_SUBDIR = "Ground truth"
_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def find_split_dir(root: str | Path, split: str) -> Path:
    """Locate ``<root>/.../<split>/Original``'s parent, tolerating one extra nesting level."""
    root = Path(root)
    candidates = [root / split] + sorted(p for p in root.glob(f"*/{split}") if p.is_dir())
    for cand in candidates:
        if (cand / IMAGE_SUBDIR).is_dir() and (cand / MASK_SUBDIR).is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find '{split}/{IMAGE_SUBDIR}' and '{split}/{MASK_SUBDIR}' under {root}"
    )


def to_input_tensor(img: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) -> normalised float tensor (3, H, W)."""
    arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((arr - _MEAN) / _STD)


class VesselSegmentationDataset(Dataset):
    """Image / mask pairs for one split, resized to ``image_size`` and cached in RAM."""

    def __init__(
        self,
        split_dir: str | Path,
        image_size: int = 224,
        augment: bool = False,
        cache: bool = True,
        mask_threshold: int = 127,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.augment = augment
        self.mask_threshold = mask_threshold
        image_dir, mask_dir = self.split_dir / IMAGE_SUBDIR, self.split_dir / MASK_SUBDIR
        self.names = sorted(p.name for p in image_dir.glob("*.png"))
        missing = [n for n in self.names if not (mask_dir / n).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images have no mask, e.g. {missing[:3]}")
        if not self.names:
            raise FileNotFoundError(f"No PNG images in {image_dir}")
        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if cache:
            self._cache = [self._load(n) for n in self.names]

    def _load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(self.split_dir / IMAGE_SUBDIR / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.split_dir / MASK_SUBDIR / name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Could not read image/mask pair {name}")
        size = (self.image_size, self.image_size)
        # Linear (not area) resize keeps thin vessels high-contrast at 224 px; area averaging
        # over ~9x9 blocks washes 1-3 px capillaries into the background.
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return img, (mask > self.mask_threshold).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.names)

    def pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw (RGB uint8 image, {0,1} uint8 mask) pair without augmentation."""
        if self._cache is not None:
            return self._cache[idx]
        return self._load(self.names[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.pair(idx)
        if self.augment:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                img, mask = img[::-1, :], mask[::-1, :]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        return to_input_tensor(img), torch.from_numpy(mask.astype(np.int64))


class SegmentationView(Dataset):
    """A subset of a ``VesselSegmentationDataset`` with its own augmentation flag.

    Lets train and validation views share one decoded cache.
    """

    def __init__(self, base: VesselSegmentationDataset, indices: Sequence[int], augment: bool):
        self.base, self.indices, self.augment = base, list(indices), augment

    def __len__(self) -> int:
        return len(self.indices)

    def pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.base.pair(self.indices[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        previous = self.base.augment
        self.base.augment = self.augment
        try:
            return self.base[self.indices[idx]]
        finally:
            self.base.augment = previous


def split_train_val(
    base: VesselSegmentationDataset, val_fraction: float = 0.2, seed: int = 42, augment: bool = True
) -> tuple[SegmentationView, SegmentationView]:
    """Seeded random train / validation split of one dataset (augmentation on train only)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(base))
    n_val = int(round(len(base) * val_fraction))
    val_idx, train_idx = sorted(order[:n_val].tolist()), sorted(order[n_val:].tolist())
    return SegmentationView(base, train_idx, augment), SegmentationView(base, val_idx, False)
