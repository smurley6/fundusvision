"""Matplotlib figures with one consistent look (saved as PNG, never shown)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")

COLORS = {
    "normal": "#2a9d8f",
    "abnormal": "#e76f51",
    "train": "#264653",
    "val": "#e9c46a",
    "muted": "#8d99ae",
}
_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
}


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(
    history: list[dict],
    panels: Sequence[tuple[str, Sequence[tuple[str, str]], bool]],
    path: str | Path,
) -> Path:
    """Generic learning-curve grid.

    ``panels`` is a list of ``(title, [(history_key, legend_label), ...], log_y)``.
    """
    epochs = [h["epoch"] for h in history]
    n = len(panels)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
        axes = np.atleast_1d(axes)
        for ax, (title, series, log_y) in zip(axes, panels, strict=True):
            for key, label in series:
                color = (
                    COLORS["train"] if "train" in key else COLORS["val"] if "val" in key else None
                )
                ax.plot(
                    epochs,
                    [h.get(key, np.nan) for h in history],
                    marker="o",
                    ms=3,
                    label=label,
                    color=color,
                )
            best = [h["epoch"] for h in history if h.get("best")]
            if best:
                ax.axvline(
                    best[-1], color=COLORS["muted"], ls="--", lw=1, label=f"best epoch {best[-1]}"
                )
            ax.set_title(title)
            ax.set_xlabel("epoch")
            if log_y:
                ax.set_yscale("log")
            ax.legend()
    return _save(fig, path)


def plot_roc_pr(
    y_true: np.ndarray, prob: np.ndarray, path: str | Path, title: str = "Test set"
) -> Path:
    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    auc, ap = roc_auc_score(y_true, prob), average_precision_score(y_true, prob)
    prevalence = float(np.mean(y_true))
    with plt.rc_context(_STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.8))
        a.plot(fpr, tpr, color=COLORS["abnormal"], lw=2, label=f"ROC AUC = {auc:.3f}")
        a.plot([0, 1], [0, 1], color=COLORS["muted"], ls="--", lw=1)
        a.set(
            xlabel="False positive rate (1 - specificity)",
            ylabel="True positive rate (sensitivity)",
            title=f"ROC - {title}",
        )
        a.legend(loc="lower right")
        b.plot(recall, precision, color=COLORS["normal"], lw=2, label=f"PR AUC = {ap:.3f}")
        b.axhline(
            prevalence, color=COLORS["muted"], ls="--", lw=1, label=f"prevalence = {prevalence:.2f}"
        )
        b.set(
            xlabel="Recall", ylabel="Precision", title=f"Precision-Recall - {title}", ylim=(0, 1.02)
        )
        b.legend(loc="lower left")
    return _save(fig, path)


def plot_confusion(
    metrics: dict, path: str | Path, class_names: Sequence[str] = ("Normal", "Abnormal")
) -> Path:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.grid(False)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                frac = cm[i, j] / max(cm[i].sum(), 1)
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({frac:.1%})",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11,
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="Actual",
            title=f"Confusion matrix @ threshold {metrics['threshold']:.2f}",
        )
    return _save(fig, path)


def plot_probability_hist(
    y_true: np.ndarray, prob: np.ndarray, threshold: float, path: str | Path
) -> Path:
    bins = np.linspace(0, 1, 41)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(prob[y_true == 0], bins, alpha=0.75, color=COLORS["normal"], label="Normal eyes")
        ax.hist(
            prob[y_true == 1], bins, alpha=0.75, color=COLORS["abnormal"], label="Abnormal eyes"
        )
        ax.axvline(threshold, color="black", ls="--", lw=1, label=f"threshold {threshold:.2f}")
        ax.set(
            xlabel="Predicted P(Abnormal)", ylabel="Eyes", title="Score distribution by true class"
        )
        ax.legend()
    return _save(fig, path)


def plot_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    path: str | Path,
    ncols: int = 4,
    suptitle: str | None = None,
    title_colors: Sequence[str] | None = None,
) -> Path:
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), squeeze=False, layout="constrained"
        )
        for k, ax in enumerate(axes.flat):
            ax.axis("off")
            if k < n:
                ax.imshow(images[k])
                ax.set_title(
                    titles[k], fontsize=8, color=(title_colors[k] if title_colors else "black")
                )
        if suptitle:
            fig.suptitle(suptitle, fontweight="bold")
    return _save(fig, path)


def plot_segmentation_examples(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    preds: Sequence[np.ndarray],
    dices: Sequence[float],
    path: str | Path,
) -> Path:
    """Rows of fundus image / ground truth / prediction / overlay (TP white, FN red, FP cyan)."""
    n = len(images)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(n, 4, figsize=(10.4, 2.7 * n), squeeze=False, layout="constrained")
        for i in range(n):
            gt, pr = masks[i].astype(bool), preds[i].astype(bool)
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            overlay[gt & pr] = (255, 255, 255)
            overlay[gt & ~pr] = (230, 57, 70)
            overlay[~gt & pr] = (0, 200, 220)
            panels = [images[i], gt, pr, overlay]
            titles = [
                "Fundus image",
                "Ground truth",
                f"Prediction (Dice {dices[i]:.2f})",
                "TP white | FN red | FP cyan",
            ]
            for ax, panel, title in zip(axes[i], panels, titles, strict=True):
                ax.axis("off")
                ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
                if i == 0:
                    ax.set_title(title, fontsize=9)
    return _save(fig, path)


def plot_segmentation_curves(history: list[dict], path: str | Path) -> Path:
    return plot_training_curves(
        history,
        [
            ("Cross-entropy loss", [("train_loss", "train"), ("val_loss", "validation")], False),
            ("Dice (vessel)", [("train_dice", "train"), ("val_dice", "validation")], False),
            ("Validation IoU", [("val_iou", "validation")], False),
        ],
        path,
    )


def plot_classifier_curves(history: list[dict], path: str | Path) -> Path:
    return plot_training_curves(
        history,
        [
            ("Weighted BCE loss", [("train_loss", "train"), ("val_loss", "validation")], False),
            ("Accuracy @ 0.5", [("train_acc", "train"), ("val_acc", "validation")], False),
            ("Validation ROC AUC", [("val_auc", "validation")], False),
            ("Learning rate", [("lr", "lr")], True),
        ],
        path,
    )
