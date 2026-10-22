"""ResNet transfer-learning classifier for Normal vs Abnormal fundus screening."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

_BACKBONES = {
    "resnet18": (models.resnet18, "ResNet18_Weights", 512),
    "resnet34": (models.resnet34, "ResNet34_Weights", 512),
    "resnet50": (models.resnet50, "ResNet50_Weights", 2048),
}


class FundusClassifier(nn.Module):
    """ImageNet-pretrained ResNet backbone + a 3-layer MLP head emitting one logit.

    ``forward`` returns raw logits of shape ``(N,)``; pair it with
    ``BCEWithLogitsLoss`` and apply ``sigmoid`` to obtain P(Abnormal).
    """

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(_BACKBONES)}, got {backbone!r}")
        builder, weights_name, features = _BACKBONES[backbone]
        weights = getattr(models, weights_name).IMAGENET1K_V1 if pretrained else None
        net = builder(weights=weights)
        # conv1 ... layer4, avgpool (drop the ImageNet fc layer)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )
        self.backbone_name = backbone

    @property
    def feature_layer(self) -> nn.Module:
        """Last convolutional stage (``layer4``), the usual Grad-CAM target."""
        return self.backbone[7]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """P(Abnormal) for a batch, shape ``(N,)``."""
        return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))
