"""Kaggle dataset download helpers.

Requires a Kaggle API token at ``~/.kaggle/kaggle.json`` (or ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables) and the optional ``kaggle`` dependency.
"""

from __future__ import annotations

from pathlib import Path

ODIR_SLUG = "andrewmvd/ocular-disease-recognition-odir5k"
VESSEL_SLUG = "nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation"

DEFAULT_ODIR_DIR = Path("data/odir5k")
DEFAULT_VESSEL_DIR = Path("data/vessel_segmentation")


def download_kaggle_dataset(slug: str, dest: str | Path) -> Path:
    """Download and unzip a Kaggle dataset into ``dest`` (created if missing)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "pip install 'fundusvision[download]' to enable Kaggle downloads"
        ) from exc
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    return dest


def download_odir(dest: str | Path = DEFAULT_ODIR_DIR) -> Path:
    """ODIR-5K (~1.9 GB unzipped): ``full_df.csv`` + ``preprocessed_images/``."""
    return download_kaggle_dataset(ODIR_SLUG, dest)


def download_vessel(dest: str | Path = DEFAULT_VESSEL_DIR) -> Path:
    """Fundus vessel segmentation set (~1.6 GB): ``train/`` and ``test/`` with masks."""
    return download_kaggle_dataset(VESSEL_SLUG, dest)
