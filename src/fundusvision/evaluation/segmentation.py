"""Segmentation metrics (pixel accuracy, IoU, Dice) and run evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fundusvision.data.vessel import VesselSegmentationDataset, find_split_dir
from fundusvision.device import get_device
from fundusvision.evaluation.plots import plot_segmentation_curves, plot_segmentation_examples
from fundusvision.models.unet import UNet
from fundusvision.training.common import load_json, load_weights, save_json


class SegmentationMeter:
    """Accumulate binary confusion counts for the vessel class over many batches.

    ``compute`` returns *micro* metrics (pooled over all pixels), matching how the
    original notebook reported IoU and Dice, plus the mean per-image Dice.
    """

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self.per_image_dice: list[float] = []

    @staticmethod
    def dice_from_counts(tp: int, fp: int, fn: int) -> float:
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = pred.bool(), target.bool()
        tp = (pred & target).flatten(1).sum(1)
        fp = (pred & ~target).flatten(1).sum(1)
        fn = (~pred & target).flatten(1).sum(1)
        tn = (~pred & ~target).flatten(1).sum(1)
        self.tp += int(tp.sum())
        self.fp += int(fp.sum())
        self.fn += int(fn.sum())
        self.tn += int(tn.sum())
        for a, b, c in zip(tp.tolist(), fp.tolist(), fn.tolist(), strict=True):
            self.per_image_dice.append(self.dice_from_counts(a, b, c))

    def compute(self) -> dict:
        total = self.tp + self.fp + self.fn + self.tn
        union = self.tp + self.fp + self.fn
        return {
            "pixel_accuracy": (self.tp + self.tn) / total if total else 0.0,
            "iou": self.tp / union if union else 1.0,
            "dice": self.dice_from_counts(self.tp, self.fp, self.fn),
            "precision": self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0,
            "recall": self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0,
            "mean_image_dice": float(np.mean(self.per_image_dice)) if self.per_image_dice else 0.0,
            "images": len(self.per_image_dice),
        }


def load_run(run_dir: str | Path, device: torch.device | None = None) -> tuple[UNet, dict]:
    run_dir = Path(run_dir)
    device = device or get_device()
    cfg = load_json(run_dir / "config.json")["config"]
    model = UNet()
    load_weights(model, run_dir / "best_model.pt", device)
    return model.eval(), cfg


@torch.no_grad()
def evaluate_segmentation(
    run_dir: str | Path,
    data_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 8,
    num_examples: int = 4,
) -> dict:
    """Score the held-out test split and save ``metrics.json`` plus example figures."""
    run_dir = Path(run_dir)
    figures_dir = Path(figures_dir) if figures_dir else run_dir / "figures"
    dev = get_device(device)
    model, cfg = load_run(run_dir, dev)
    data_dir = Path(data_dir) if data_dir else Path(cfg["data_dir"])
    test_ds = VesselSegmentationDataset(
        find_split_dir(data_dir, "test"), cfg["image_size"], cache=False
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    meter, preds = SegmentationMeter(), []
    for x, y in loader:
        p = model.predict_mask(x.to(dev)).cpu()
        meter.update(p, y)
        preds.append(p.numpy().astype(np.uint8))
    pred_masks = np.concatenate(preds)
    metrics = meter.compute()
    report = {"run_dir": str(run_dir), "test": metrics, "test_images": len(test_ds)}
    save_json(report, run_dir / "metrics.json")

    plot_segmentation_curves(
        load_json(run_dir / "history.json"), figures_dir / "segmentation_training_curves.png"
    )
    # Show a spread of quality: best, median-ish and worst per-image Dice.
    dice = np.array(meter.per_image_dice)
    order = np.argsort(dice)
    picks = sorted(
        {
            int(order[-1]),
            int(order[len(order) // 2]),
            int(order[0]),
            *order[-num_examples:].tolist(),
        }
    )[:num_examples]
    pairs = [test_ds.pair(i) for i in picks]
    plot_segmentation_examples(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        [pred_masks[i] for i in picks],
        [dice[i] for i in picks],
        figures_dir / "segmentation_examples.png",
    )
    print(
        f"test: pixel acc {metrics['pixel_accuracy']:.4f} | IoU {metrics['iou']:.4f}"
        f" | Dice {metrics['dice']:.4f}"
        f" | mean per-image Dice {metrics['mean_image_dice']:.4f} ({metrics['images']} images)"
    )
    return report
