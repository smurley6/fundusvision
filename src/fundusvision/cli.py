"""Command-line interface: ``fundusvision <command> [options]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fundusvision", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser(
        "download", help="download a dataset from Kaggle (needs ~/.kaggle/kaggle.json)"
    )
    d.add_argument("dataset", choices=["odir", "vessel"])
    d.add_argument(
        "--dest",
        default=None,
        help="target directory (default: data/odir5k or data/vessel_segmentation)",
    )

    tc = sub.add_parser("train-classifier", help="train the ResNet Normal-vs-Abnormal classifier")
    tc.add_argument("--data-dir", default="data/odir5k")
    tc.add_argument("--output-dir", default="results/classifier")
    tc.add_argument("--backbone", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    tc.add_argument(
        "--no-pretrained", action="store_true", help="random init instead of ImageNet weights"
    )
    tc.add_argument("--epochs", type=int, default=50)
    tc.add_argument("--patience", type=int, default=10)
    tc.add_argument("--batch-size", type=int, default=16)
    tc.add_argument("--lr", type=float, default=1e-4)
    tc.add_argument("--weight-decay", type=float, default=1e-4)
    tc.add_argument("--dropout", type=float, default=0.5)
    tc.add_argument("--image-size", type=int, default=224)
    tc.add_argument("--num-workers", type=int, default=0)
    tc.add_argument(
        "--no-cache", action="store_true", help="decode images on the fly instead of caching in RAM"
    )
    tc.add_argument("--limit", type=int, default=None, help="use only N images (smoke tests)")
    tc.add_argument("--no-progress", action="store_true", help="disable tqdm bars (log files)")
    _add_common(tc)

    ec = sub.add_parser(
        "eval-classifier", help="evaluate a classifier run on validation/test and save figures"
    )
    ec.add_argument("--run-dir", default="results/classifier")
    ec.add_argument("--figures-dir", default="docs/figures")
    ec.add_argument("--batch-size", type=int, default=32)
    ec.add_argument("--num-examples", type=int, default=12)
    _add_common(ec)

    ex = sub.add_parser("explain", help="Grad-CAM overlays for a classifier run")
    ex.add_argument("--run-dir", default="results/classifier")
    ex.add_argument("--out", default="docs/figures/classifier_gradcam.png")
    ex.add_argument("--num-examples", type=int, default=8)
    _add_common(ex)

    ts = sub.add_parser("train-segmentation", help="train the U-Net vessel segmentation model")
    ts.add_argument("--data-dir", default="data/vessel_segmentation")
    ts.add_argument("--output-dir", default="results/segmentation")
    ts.add_argument("--epochs", type=int, default=20)
    ts.add_argument("--batch-size", type=int, default=8)
    ts.add_argument("--lr", type=float, default=1e-4)
    ts.add_argument("--image-size", type=int, default=224)
    ts.add_argument("--val-fraction", type=float, default=0.2)
    ts.add_argument(
        "--augment", action="store_true", help="random flip augmentation (off by default)"
    )
    ts.add_argument("--cosine", action="store_true", help="cosine LR annealing (off by default)")
    ts.add_argument(
        "--select-on", default="dice", choices=["dice", "loss"], help="checkpoint selection metric"
    )
    ts.add_argument("--num-workers", type=int, default=0)
    ts.add_argument("--no-cache", action="store_true")
    ts.add_argument("--no-progress", action="store_true")
    _add_common(ts)

    es = sub.add_parser("eval-segmentation", help="evaluate a segmentation run on the test split")
    es.add_argument("--run-dir", default="results/segmentation")
    es.add_argument(
        "--data-dir", default=None, help="override the data dir stored in the run config"
    )
    es.add_argument("--figures-dir", default="docs/figures")
    es.add_argument("--batch-size", type=int, default=8)
    es.add_argument("--num-examples", type=int, default=4)
    _add_common(es)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        from fundusvision.data.download import (
            DEFAULT_ODIR_DIR,
            DEFAULT_VESSEL_DIR,
            download_odir,
            download_vessel,
        )

        if args.dataset == "odir":
            print(download_odir(args.dest or DEFAULT_ODIR_DIR))
        else:
            print(download_vessel(args.dest or DEFAULT_VESSEL_DIR))
        return 0

    if args.command == "train-classifier":
        from fundusvision.training.classifier import ClassifierConfig, train_classifier

        cfg = ClassifierConfig(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            dropout=args.dropout,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            num_workers=args.num_workers,
            cache=not args.no_cache,
            device=args.device,
            limit=args.limit,
            progress=not args.no_progress,
        )
        print(json.dumps(train_classifier(cfg), indent=2))
        return 0

    if args.command == "eval-classifier":
        from fundusvision.evaluation.classifier import evaluate_classifier

        evaluate_classifier(
            args.run_dir, args.figures_dir, args.device, args.batch_size, args.num_examples
        )
        return 0

    if args.command == "explain":
        from fundusvision.explain.gradcam import make_gradcam_figure

        print(
            make_gradcam_figure(args.run_dir, args.out, args.num_examples, args.device, args.seed)
        )
        return 0

    if args.command == "train-segmentation":
        from fundusvision.training.segmentation import SegmentationConfig, train_segmentation

        cfg = SegmentationConfig(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            val_fraction=args.val_fraction,
            augment=args.augment,
            cosine=args.cosine,
            select_on=args.select_on,
            seed=args.seed,
            num_workers=args.num_workers,
            cache=not args.no_cache,
            device=args.device,
            progress=not args.no_progress,
        )
        print(json.dumps(train_segmentation(cfg), indent=2))
        return 0

    if args.command == "eval-segmentation":
        from fundusvision.evaluation.segmentation import evaluate_segmentation

        evaluate_segmentation(
            args.run_dir,
            args.data_dir,
            args.figures_dir,
            args.device,
            args.batch_size,
            args.num_examples,
        )
        return 0
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
