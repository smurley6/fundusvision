"""Synthetic fixtures: tiny fake ODIR-5K and vessel datasets so tests need no downloads."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


def _fake_fundus(rng: np.random.Generator, size: int, abnormal: bool) -> np.ndarray:
    """Dark disc on black, with a bright blob when 'abnormal' so a model can learn something."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 2, (120, 60, 30), -1)
    if abnormal:
        c = (
            int(rng.integers(size // 4, 3 * size // 4)),
            int(rng.integers(size // 4, 3 * size // 4)),
        )
        cv2.circle(img, c, size // 8, (240, 230, 200), -1)
    noise = rng.integers(0, 20, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


@pytest.fixture(scope="session")
def odir_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``full_df.csv`` + ``preprocessed_images/`` with 24 patients (48 eyes) at 64 px."""
    root = tmp_path_factory.mktemp("odir5k")
    images = root / "preprocessed_images"
    images.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    keywords = [
        "normal fundus",
        "cataract",
        "normal fundus，lens dust",
        "moderate non proliferative retinopathy",
    ]
    for pid in range(24):
        kw_l, kw_r = keywords[pid % 4], keywords[(pid + 1) % 4]
        for side, kw in (("left", kw_l), ("right", kw_r)):
            name = f"{pid}_{side}.jpg"
            cv2.imwrite(str(images / name), _fake_fundus(rng, 64, "normal fundus" not in kw))
        # Kaggle's full_df has one row per image, each carrying both eyes' metadata.
        for side in ("left", "right"):
            rows.append(
                {
                    "ID": pid,
                    "Patient Age": 50,
                    "Patient Sex": "Female",
                    "Left-Fundus": f"{pid}_left.jpg",
                    "Right-Fundus": f"{pid}_right.jpg",
                    "Left-Diagnostic Keywords": kw_l,
                    "Right-Diagnostic Keywords": kw_r,
                    "filename": f"{pid}_{side}.jpg",
                }
            )
    pd.DataFrame(rows).to_csv(root / "full_df.csv", index=False)
    return root


@pytest.fixture(scope="session")
def vessel_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``train/`` (10 pairs) and ``test/`` (4 pairs) of 64 px images with line-shaped masks."""
    root = tmp_path_factory.mktemp("vessel")
    rng = np.random.default_rng(1)
    for split, n in (("train", 10), ("test", 4)):
        (root / split / "Original").mkdir(parents=True)
        (root / split / "Ground truth").mkdir(parents=True)
        for i in range(n):
            mask = np.zeros((64, 64), dtype=np.uint8)
            for _ in range(3):
                p1 = tuple(int(v) for v in rng.integers(0, 64, 2))
                p2 = tuple(int(v) for v in rng.integers(0, 64, 2))
                cv2.line(mask, p1, p2, 255, 2)
            img = np.full((64, 64, 3), 40, dtype=np.uint8)
            img[mask > 0] = (200, 60, 60)
            cv2.imwrite(str(root / split / "Original" / f"{i}.png"), img)
            cv2.imwrite(
                str(root / split / "Ground truth" / f"{i}.png"),
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            )
    return root
