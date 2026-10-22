"""CPU smoke tests: a one-epoch train + evaluate for each task on synthetic data."""

from __future__ import annotations

import json

from fundusvision.cli import main


def test_classifier_train_eval_explain(odir_dir, tmp_path):
    run = tmp_path / "clf"
    figs = tmp_path / "figs"
    assert (
        main(
            [
                "train-classifier",
                "--data-dir",
                str(odir_dir),
                "--output-dir",
                str(run),
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--image-size",
                "64",
                "--no-pretrained",
                "--device",
                "cpu",
                "--no-progress",
            ]
        )
        == 0
    )
    for name in ("best_model.pt", "history.json", "split.csv", "config.json"):
        assert (run / name).exists()
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 1 and history[0]["best"] is True

    assert (
        main(
            [
                "eval-classifier",
                "--run-dir",
                str(run),
                "--figures-dir",
                str(figs),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report = json.loads((run / "metrics.json").read_text())
    assert set(report["test"]) == {"default", "val_f1", "val_youden"}
    assert report["test"]["val_f1"]["n"] > 0
    assert (figs / "classifier_roc_pr.png").exists()
    assert (figs / "classifier_confusion_matrix.png").exists()

    out = figs / "gradcam.png"
    assert (
        main(
            [
                "explain",
                "--run-dir",
                str(run),
                "--out",
                str(out),
                "--num-examples",
                "4",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert out.exists()


def test_segmentation_train_eval(vessel_dir, tmp_path):
    run = tmp_path / "seg"
    figs = tmp_path / "figs"
    assert (
        main(
            [
                "train-segmentation",
                "--data-dir",
                str(vessel_dir),
                "--output-dir",
                str(run),
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--image-size",
                "64",
                "--device",
                "cpu",
                "--no-progress",
            ]
        )
        == 0
    )
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2 and any(h["best"] for h in history)
    assert (run / "best_model.pt").exists()

    assert (
        main(
            [
                "eval-segmentation",
                "--run-dir",
                str(run),
                "--figures-dir",
                str(figs),
                "--device",
                "cpu",
                "--num-examples",
                "2",
            ]
        )
        == 0
    )
    report = json.loads((run / "metrics.json").read_text())
    assert report["test_images"] == 4
    assert 0 <= report["test"]["dice"] <= 1
    assert (figs / "segmentation_examples.png").exists()
