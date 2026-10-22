"""ODIR-5K data pipeline: per-eye labels, patient-grouped splits, dataset and transforms.

Label and split logic
---------------------
Each row of the Kaggle ``full_df.csv`` carries *both* eyes of a patient (filename and
free-text diagnostic keywords per eye). The pipeline here:

1. Melts the table to one row per **eye** (physical image), so every image is used once.
2. Labels each eye from **its own** keywords: *Normal* (0) only when the sole meaningful
   token is ``"normal fundus"`` (capture artefacts such as ``"lens dust"`` are ignored);
   otherwise *Abnormal* (1).
3. Splits **by patient** with ``StratifiedGroupKFold`` so that no patient, and hence no
   image, appears in more than one of train / validation / test.

An earlier version of this project split at the row level with each image duplicated
~1.95x, which leaked test images into training. ``assert_no_leakage`` guards against that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

NORMAL_TOKEN = "normal fundus"
ARTIFACT_TOKENS = frozenset({"lens dust", "low image quality"})
CLASS_NAMES = ("Normal", "Abnormal")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_KEYWORD_SEP = re.compile(r"[，,]")  # ODIR mixes ASCII and full-width commas


def keyword_to_label(keywords: object) -> int:
    """Map one eye's diagnostic keyword string to ``0`` (Normal) or ``1`` (Abnormal).

    An eye is Normal only when every meaningful token equals ``"normal fundus"``.
    Non-pathological capture artefacts (``ARTIFACT_TOKENS``) are ignored; an eye whose
    only tokens are artefacts, or that has no keywords at all, is treated as Abnormal.
    """
    text = (
        ""
        if keywords is None or (isinstance(keywords, float) and np.isnan(keywords))
        else str(keywords)
    )
    tokens = [t.strip().lower() for t in _KEYWORD_SEP.split(text) if t.strip()]
    meaningful = [t for t in tokens if t not in ARTIFACT_TOKENS]
    if meaningful and all(t == NORMAL_TOKEN for t in meaningful):
        return 0
    return 1


def build_eye_table(csv_path: str | Path, image_dir: str | Path) -> pd.DataFrame:
    """Return one row per physical fundus image that exists on disk.

    Columns: ``patient_id, filename, side, keywords, path, label``.
    """
    csv_path, image_dir = Path(csv_path), Path(image_dir)
    df = pd.read_csv(csv_path)
    frames = []
    for side in ("Left", "Right"):
        part = df[["ID", f"{side}-Fundus", f"{side}-Diagnostic Keywords"]].rename(
            columns={
                "ID": "patient_id",
                f"{side}-Fundus": "filename",
                f"{side}-Diagnostic Keywords": "keywords",
            }
        )
        part["side"] = side.lower()
        frames.append(part)
    eyes = pd.concat(frames, ignore_index=True)

    on_disk = {p.name for p in image_dir.iterdir()} if image_dir.is_dir() else set()
    eyes = eyes[eyes["filename"].isin(on_disk)].copy()
    if eyes.empty:
        raise FileNotFoundError(f"No images referenced by {csv_path} were found in {image_dir}")

    conflicts = eyes.groupby("filename")["keywords"].nunique()
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()[:5]
        raise ValueError(f"Conflicting keywords for the same image: {bad}")

    eyes = eyes.drop_duplicates(subset="filename").reset_index(drop=True)
    eyes["path"] = eyes["filename"].map(lambda f: str(image_dir / f))
    eyes["label"] = eyes["keywords"].map(keyword_to_label).astype(int)
    return eyes[["patient_id", "filename", "side", "keywords", "path", "label"]]


@dataclass(frozen=True)
class Split:
    """Train / validation / test partitions of an eye table."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def as_frame(self) -> pd.DataFrame:
        """Single table with a ``split`` column, handy for saving alongside a run."""
        parts = [p.assign(split=name) for name, p in self.items()]
        return pd.concat(parts, ignore_index=True)

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [("train", self.train), ("val", self.val), ("test", self.test)]

    def summary(self) -> str:
        lines = []
        for name, part in self.items():
            lines.append(
                f"{name:5s}: {len(part):5d} images | {part['patient_id'].nunique():5d} patients"
                f" | {100 * part['label'].mean():5.1f}% abnormal"
            )
        return "\n".join(lines)


def assert_no_leakage(split: Split) -> None:
    """Raise ``ValueError`` if any image path or patient appears in two partitions."""
    for key in ("path", "patient_id"):
        sets = {name: set(part[key]) for name, part in split.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f"{key} leakage between {a} and {b}: {len(overlap)} shared")


def patient_grouped_split(
    table: pd.DataFrame, seed: int = 42, test_folds: int = 7, val_folds: int = 6
) -> Split:
    """Patient-grouped, label-stratified split (~71% / 14% / 14% by default).

    The test set is one fold of a ``test_folds``-way grouped split; the validation set is
    one fold of a ``val_folds``-way grouped split of the remainder.
    """
    labels, groups = table["label"].to_numpy(), table["patient_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    tv_idx, te_idx = next(outer.split(np.zeros(len(table)), labels, groups))
    trainval = table.iloc[tv_idx].reset_index(drop=True)
    test = table.iloc[te_idx].reset_index(drop=True)

    inner = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(
        inner.split(
            np.zeros(len(trainval)),
            trainval["label"].to_numpy(),
            trainval["patient_id"].to_numpy(),
        )
    )
    split = Split(
        train=trainval.iloc[tr_idx].reset_index(drop=True),
        val=trainval.iloc[va_idx].reset_index(drop=True),
        test=test,
    )
    assert_no_leakage(split)
    if len(split.train) + len(split.val) + len(split.test) != len(table):
        raise ValueError("split sizes do not add up to the table size")
    return split


def build_transforms(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    """Torchvision pipeline from an RGB uint8 array to a normalised tensor.

    Augmentation (training only): flips, small rotations / affine jitter, colour jitter
    and occasional Gaussian blur. Validation and test images are only resized.
    """
    ops: list = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=(0.7, 1.3), contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def load_rgb(path: str | Path, size: int | None = None) -> np.ndarray:
    """Read an image as an RGB uint8 array, optionally resized to ``size`` x ``size``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None and img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


class FundusDataset(Dataset):
    """Fundus images + binary labels from an eye table (``path`` and ``label`` columns).

    With ``cache=True`` every image is decoded and resized once up front, which makes
    training I/O-free afterwards (6,392 images at 224 px is ~1 GB of RAM).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: transforms.Compose | None = None,
        cache: bool = False,
        cache_size: int = 224,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.paths = self.frame["path"].tolist()
        self.labels = self.frame["label"].to_numpy(dtype=np.int64)
        self._cache: list[np.ndarray] | None = None
        if cache:
            self._cache = [load_rgb(p, cache_size) for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def image(self, idx: int) -> np.ndarray:
        """Raw RGB uint8 image (cached if available)."""
        if self._cache is not None:
            return self._cache[idx]
        return load_rgb(self.paths[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.image(idx)
        x = self.transform(img) if self.transform is not None else torch.from_numpy(img)
        return x, torch.tensor(float(self.labels[idx]), dtype=torch.float32)
