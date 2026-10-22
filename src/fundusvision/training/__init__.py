"""Training loops."""

from fundusvision.training.classifier import ClassifierConfig, train_classifier
from fundusvision.training.segmentation import SegmentationConfig, train_segmentation

__all__ = ["ClassifierConfig", "SegmentationConfig", "train_classifier", "train_segmentation"]
