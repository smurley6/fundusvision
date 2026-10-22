"""Evaluate a trained classifier run: metrics JSON + figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fundusvision.data.odir import CLASS_NAMES, FundusDataset, build_transforms
from fundusvision.device import get_device
from fundusvision.evaluation.metrics import binary_metrics, find_threshold
from fundusvision.evaluation.plots import (
    COLORS,
    plot_classifier_curves,
    plot_confusion,
    plot_image_grid,
    plot_probability_hist,
    plot_roc_pr,
)
from fundusvision.models.classifier import FundusClassifier
from fundusvision.training.common import load_json, load_weights, save_json


def load_run(
    run_dir: str | Path, device: torch.device | None = None
) -> tuple[FundusClassifier, dict, pd.DataFrame]:
    """Rebuild the model and split of a finished run from its ``run_dir``."""
    run_dir = Path(run_dir)
    device = device or get_device()
    cfg = load_json(run_dir / "config.json")["config"]
    model = FundusClassifier(cfg["backbone"], pretrained=False, dropout=cfg["dropout"])
    load_weights(model, run_dir / "best_model.pt", device)
    model.eval()
    split = pd.read_csv(run_dir / "split.csv")
    return model, cfg, split


@torch.no_grad()
def predict(
    model: FundusClassifier, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(P(Abnormal), labels)`` for every item in ``loader``."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(model.predict_proba(x.to(device)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int)


def evaluate_classifier(
    run_dir: str | Path,
    figures_dir: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 32,
    num_examples: int = 12,
) -> dict:
    """Select thresholds on validation, report on test, and save figures.

    Writes ``metrics.json`` into ``run_dir`` and PNGs into ``figures_dir``.
    """
    run_dir = Path(run_dir)
    figures_dir = Path(figures_dir) if figures_dir else run_dir / "figures"
    dev = get_device(device)
    model, cfg, split = load_run(run_dir, dev)
    transform = build_transforms(cfg["image_size"], augment=False)

    def loader_for(part: str) -> tuple[FundusDataset, DataLoader]:
        ds = FundusDataset(split[split["split"] == part], transform=transform)
        return ds, DataLoader(ds, batch_size=batch_size, shuffle=False)

    _, val_loader = loader_for("val")
    test_ds, test_loader = loader_for("test")
    val_p, val_y = predict(model, val_loader, dev)
    test_p, test_y = predict(model, test_loader, dev)

    thr_f1, _ = find_threshold(val_y, val_p, "f1")
    thr_youden, _ = find_threshold(val_y, val_p, "youden")
    report = {
        "run_dir": str(run_dir),
        "backbone": cfg["backbone"],
        "thresholds": {"default": 0.5, "val_f1": thr_f1, "val_youden": thr_youden},
        "validation": {"default": binary_metrics(val_y, val_p, 0.5)},
        "test": {
            "default": binary_metrics(test_y, test_p, 0.5),
            "val_f1": binary_metrics(test_y, test_p, thr_f1),
            "val_youden": binary_metrics(test_y, test_p, thr_youden),
        },
    }
    save_json(report, run_dir / "metrics.json")
    np.savez_compressed(run_dir / "test_predictions.npz", prob=test_p, label=test_y)

    headline = report["test"]["val_youden"]  # balanced sens/spec: the screening operating point
    history = load_json(run_dir / "history.json")
    plot_classifier_curves(history, figures_dir / "classifier_training_curves.png")
    plot_roc_pr(test_y, test_p, figures_dir / "classifier_roc_pr.png")
    plot_confusion(headline, figures_dir / "classifier_confusion_matrix.png", CLASS_NAMES)
    plot_probability_hist(
        test_y, test_p, thr_youden, figures_dir / "classifier_score_distribution.png"
    )

    wrong = np.where((test_p >= thr_youden).astype(int) != test_y)[0]
    order = wrong[np.argsort(-np.abs(test_p[wrong] - thr_youden))][
        :num_examples
    ]  # most confident errors
    if len(order):
        images = [test_ds.image(int(i)) for i in order]
        titles = [f"true {CLASS_NAMES[test_y[i]]} | P(abn) {test_p[i]:.2f}" for i in order]
        colors = [COLORS["abnormal"] if test_y[i] == 1 else COLORS["normal"] for i in order]
        plot_image_grid(
            images,
            titles,
            figures_dir / "classifier_misclassified.png",
            ncols=4,
            suptitle="Most confident test-set errors",
            title_colors=colors,
        )
    print(
        f"test @ val-Youden threshold {thr_youden:.3f}: acc {headline['accuracy']:.3f}"
        f" | sens {headline['recall']:.3f} | spec {headline['specificity']:.3f}"
        f" | F1 {headline['f1']:.3f} | ROC AUC {headline['roc_auc']:.3f}"
        f" | PR AUC {headline['pr_auc']:.3f}"
    )
    return report
