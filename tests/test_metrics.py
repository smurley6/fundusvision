import numpy as np
import torch

from fundusvision.evaluation.metrics import binary_metrics, find_threshold
from fundusvision.evaluation.segmentation import SegmentationMeter


def test_binary_metrics_counts_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.6, 0.3])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 2, 1)
    assert np.isclose(m["recall"], 2 / 3) and np.isclose(m["specificity"], 2 / 3)
    assert np.isclose(m["accuracy"], 4 / 6) and np.isclose(m["f1"], 2 / 3)
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1


def test_find_threshold_separates_perfectly_separable_scores():
    y = np.array([0] * 10 + [1] * 10)
    p = np.concatenate([np.linspace(0.0, 0.4, 10), np.linspace(0.6, 1.0, 10)])
    thr_f1, f1 = find_threshold(y, p, "f1")
    thr_j, j = find_threshold(y, p, "youden")
    assert 0.4 < thr_f1 <= 0.6 and np.isclose(f1, 1.0)
    assert 0.4 < thr_j <= 0.6 and np.isclose(j, 1.0)


def test_segmentation_meter_micro_and_per_image():
    pred = torch.tensor([[[1, 1], [0, 0]], [[1, 0], [0, 0]]])
    target = torch.tensor([[[1, 0], [0, 0]], [[1, 1], [0, 0]]])
    meter = SegmentationMeter()
    meter.update(pred, target)
    m = meter.compute()
    # pooled: tp=2, fp=1, fn=1, tn=4
    assert np.isclose(m["iou"], 2 / 4) and np.isclose(m["dice"], 4 / 6)
    assert np.isclose(m["pixel_accuracy"], 6 / 8)
    assert m["images"] == 2 and np.isclose(m["mean_image_dice"], (2 / 3 + 2 / 3) / 2)
