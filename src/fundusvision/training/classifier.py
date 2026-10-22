"""Train the Normal-vs-Abnormal fundus classifier."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fundusvision.data.odir import (
    FundusDataset,
    Split,
    build_eye_table,
    build_transforms,
    patient_grouped_split,
)
from fundusvision.device import get_device, seed_everything
from fundusvision.models.classifier import FundusClassifier, count_parameters
from fundusvision.training.common import EarlyStopping, save_json, save_weights


@dataclass
class ClassifierConfig:
    """Hyper-parameters and paths for a classifier training run."""

    data_dir: Path = Path("data/odir5k")
    output_dir: Path = Path("results/classifier")
    backbone: str = "resnet34"
    pretrained: bool = True
    dropout: float = 0.5
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_t0: int = 10
    scheduler_tmult: int = 2
    eta_min: float = 1e-6
    seed: int = 42
    num_workers: int = 0
    cache: bool = True
    device: str | None = None
    limit: int | None = None  # subsample images (smoke tests only)
    progress: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def csv_path(self) -> Path:
        return Path(self.data_dir) / "full_df.csv"

    @property
    def image_dir(self) -> Path:
        return Path(self.data_dir) / "preprocessed_images"


def build_split(cfg: ClassifierConfig) -> Split:
    """Eye table -> patient-grouped split (optionally subsampled for smoke tests)."""
    table = build_eye_table(cfg.csv_path, cfg.image_dir)
    if cfg.limit is not None and cfg.limit < len(table):
        table = table.sample(n=cfg.limit, random_state=cfg.seed).reset_index(drop=True)
    return patient_grouped_split(table, seed=cfg.seed)


def make_loader(
    frame: pd.DataFrame, cfg: ClassifierConfig, train: bool, device: torch.device
) -> DataLoader:
    dataset = FundusDataset(
        frame,
        transform=build_transforms(cfg.image_size, augment=train),
        cache=cfg.cache,
        cache_size=cfg.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        drop_last=train and len(dataset) > cfg.batch_size,  # no size-1 batch for BatchNorm
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress: bool = False,
    desc: str = "",
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over ``loader``. Trains when an optimizer is given, else evaluates.

    Returns ``(mean loss, probabilities, labels)``.
    """
    training = optimizer is not None
    model.train(training)
    total, probs, labels = 0.0, [], []
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
            probs.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
            labels.append(y.cpu().numpy())
    probs_np, labels_np = np.concatenate(probs), np.concatenate(labels).astype(int)
    return total / max(len(labels_np), 1), probs_np, labels_np


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def train_classifier(cfg: ClassifierConfig) -> dict:
    """Full training run. Writes ``split.csv``, ``history.json``, ``config.json`` and
    ``best_model.pt`` into ``cfg.output_dir`` and returns a summary dict."""
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split = build_split(cfg)
    split.as_frame().to_csv(out / "split.csv", index=False)
    print(f"Device: {device}\n{split.summary()}")

    train_loader = make_loader(split.train, cfg, train=True, device=device)
    val_loader = make_loader(split.val, cfg, train=False, device=device)

    model = FundusClassifier(cfg.backbone, pretrained=cfg.pretrained, dropout=cfg.dropout).to(
        device
    )
    n_pos = int(split.train["label"].sum())
    n_neg = len(split.train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.scheduler_t0, T_mult=cfg.scheduler_tmult, eta_min=cfg.eta_min
    )
    stopper = EarlyStopping(cfg.patience, mode="min")
    print(
        f"Model: {cfg.backbone} ({count_parameters(model):,} params)"
        f" | pos_weight={pos_weight.item():.3f}"
    )

    history: list[dict] = []
    started = time.time()
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, train_p, train_y = run_epoch(
            model, train_loader, criterion, device, optimizer, cfg.progress, f"train {epoch + 1}"
        )
        val_loss, val_p, val_y = run_epoch(
            model, val_loader, criterion, device, None, cfg.progress, f"val {epoch + 1}"
        )
        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": float(((train_p > 0.5) == train_y).mean()),
            "val_loss": val_loss,
            "val_acc": float(((val_p > 0.5) == val_y).mean()),
            "val_f1": float(f1_score(val_y, val_p > 0.5, zero_division=0)),
            "val_auc": _safe_auc(val_y, val_p),
            "seconds": time.time() - t0,
        }
        scheduler.step()
        improved = stopper.step(val_loss, epoch + 1)
        if improved:
            save_weights(model, out / "best_model.pt")
        record["best"] = improved
        history.append(record)
        save_json(history, out / "history.json")
        print(
            f"epoch {epoch + 1:3d}/{cfg.epochs} | train loss {train_loss:.4f}"
            f" acc {record['train_acc']:.3f} | val loss {val_loss:.4f}"
            f" acc {record['val_acc']:.3f} f1 {record['val_f1']:.3f}"
            f" auc {record['val_auc']:.3f} | lr {record['lr']:.2e} | {record['seconds']:.0f}s"
            f"{'  *best*' if improved else ''}"
        )
        if stopper.should_stop:
            print(f"Early stopping: no val-loss improvement for {cfg.patience} epochs.")
            break

    summary = {
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best,
        "epochs_run": len(history),
        "train_images": len(split.train),
        "val_images": len(split.val),
        "test_images": len(split.test),
        "parameters": count_parameters(model),
        "device": str(device),
        "minutes": (time.time() - started) / 60,
    }
    save_json({"config": cfg, "summary": summary}, out / "config.json")
    print(f"Done in {summary['minutes']:.1f} min; best epoch {summary['best_epoch']}")
    return summary
