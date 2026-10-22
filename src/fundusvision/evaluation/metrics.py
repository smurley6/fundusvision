"""Binary classification metrics and decision-threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Confusion counts plus the usual screening metrics at ``threshold``.

    Threshold-free metrics (ROC AUC, PR AUC) are included for convenience.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    precision, npv = ratio(tp, tp + fp), ratio(tn, tn + fn)
    two_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": ratio(tp + tn, len(y)),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "recall": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "youden": sensitivity + specificity - 1,
        "mcc": float(matthews_corrcoef(y, pred)) if two_classes else float("nan"),
        "kappa": float(cohen_kappa_score(y, pred)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
    }


def find_threshold(y_true: np.ndarray, prob: np.ndarray, metric: str = "f1") -> tuple[float, float]:
    """Pick the decision threshold maximising ``metric`` (``"f1"`` or ``"youden"``).

    Call this on the **validation** split and apply the result to the test split; tuning on
    the test set would leak information into the reported numbers.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y, p)
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best = int(np.argmax(f1))
        return float(thresholds[best]), float(f1[best])
    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(y, p)
        j = tpr - fpr
        best = int(np.argmax(j))
        return float(min(thresholds[best], 1.0)), float(j[best])
    raise ValueError("metric must be 'f1' or 'youden'")
