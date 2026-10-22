"""Train the U-Net vessel segmentation model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fundusvision.data.vessel import VesselSegmentationDataset, find_split_dir, split_train_val
from fundusvision.device import get_device, seed_everything
from fundusvision.evaluation.segmentation import SegmentationMeter
from fundusvision.models.classifier import count_parameters
from fundusvision.models.unet import UNet
from fundusvision.training.common import EarlyStopping, save_json, save_weights


@dataclass
class SegmentationConfig:
    """Hyper-parameters and paths for a segmentation training run."""

    data_dir: Path = Path("data/vessel_segmentation")
    output_dir: Path = Path("results/segmentation")
    image_size: int = 224
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    val_fraction: float = 0.2
    augment: bool = False  # random flips; hurt this small dataset (see docs/segmentation.md)
    cosine: bool = False  # cosine annealing; the constant-LR original recipe trains better here
    select_on: str = "dice"  # checkpoint selection: 'dice' (max val Dice) or 'loss' (min val loss)
    seed: int = 42
    num_workers: int = 0
    cache: bool = True
    device: str | None = None
    progress: bool = True
    extra: dict = field(default_factory=dict)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress: bool = False,
    desc: str = "",
) -> tuple[float, SegmentationMeter]:
    """One pass over ``loader``; trains when an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    meter, total, count = SegmentationMeter(), 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False, disable=not progress):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
            meter.update(logits.detach().argmax(dim=1), y)
    return total / max(count, 1), meter


def train_segmentation(cfg: SegmentationConfig) -> dict:
    """Full training run; checkpoints the best validation Dice to ``best_model.pt``."""
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = VesselSegmentationDataset(
        find_split_dir(cfg.data_dir, "train"), cfg.image_size, cache=cfg.cache
    )
    train_view, val_view = split_train_val(base, cfg.val_fraction, cfg.seed, cfg.augment)
    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_view, cfg.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_view, cfg.batch_size, shuffle=False, **loader_kw)
    print(f"Device: {device} | train {len(train_view)} | val {len(val_view)} images")

    model = UNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.cosine
        else torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    )
    if cfg.select_on not in ("dice", "loss"):
        raise ValueError("select_on must be 'dice' or 'loss'")
    # Track the best epoch only (no early stopping): all `epochs` are always run.
    stopper = EarlyStopping(patience=cfg.epochs, mode="max" if cfg.select_on == "dice" else "min")
    print(f"Model: U-Net ({count_parameters(model):,} params)")

    history: list[dict] = []
    started = time.time()
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, train_meter = run_epoch(
            model, train_loader, criterion, device, optimizer, cfg.progress, f"train {epoch + 1}"
        )
        val_loss, val_meter = run_epoch(
            model, val_loader, criterion, device, None, cfg.progress, f"val {epoch + 1}"
        )
        val_metrics = val_meter.compute()
        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_dice": train_meter.compute()["dice"],
            "val_loss": val_loss,
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "seconds": time.time() - t0,
        }
        scheduler.step()
        criterion_value = record["val_dice"] if cfg.select_on == "dice" else val_loss
        improved = stopper.step(criterion_value, epoch + 1)
        if improved:
            save_weights(model, out / "best_model.pt")
        record["best"] = improved
        history.append(record)
        save_json(history, out / "history.json")
        print(
            f"epoch {epoch + 1:3d}/{cfg.epochs} | train loss {train_loss:.4f}"
            f" dice {record['train_dice']:.3f}"
            f" | val loss {val_loss:.4f} acc {record['val_pixel_accuracy']:.4f}"
            f" iou {record['val_iou']:.3f} dice {record['val_dice']:.3f} | lr {record['lr']:.2e}"
            f" | {record['seconds']:.0f}s{'  *best*' if improved else ''}"
        )

    summary = {
        "best_epoch": stopper.best_epoch,
        "best_val_dice": max(h["val_dice"] for h in history),
        "best_val_loss": min(h["val_loss"] for h in history),
        "selected_on": cfg.select_on,
        "epochs_run": len(history),
        "train_images": len(train_view),
        "val_images": len(val_view),
        "parameters": count_parameters(model),
        "device": str(device),
        "minutes": (time.time() - started) / 60,
    }
    save_json({"config": cfg, "summary": summary}, out / "config.json")
    print(f"Done in {summary['minutes']:.1f} min; best epoch {summary['best_epoch']}")
    return summary
