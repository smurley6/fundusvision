"""Evaluation: metrics, figures and run reports."""

from fundusvision.evaluation.classifier import evaluate_classifier
from fundusvision.evaluation.metrics import binary_metrics, find_threshold
from fundusvision.evaluation.segmentation import SegmentationMeter, evaluate_segmentation

__all__ = [
    "SegmentationMeter",
    "binary_metrics",
    "evaluate_classifier",
    "evaluate_segmentation",
    "find_threshold",
]
