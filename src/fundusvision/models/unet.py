"""U-Net for binary retinal vessel segmentation (Ronneberger et al., 2015)."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """4-level encoder/decoder with skip connections and a 1x1 output head.

    Default widths ``(64, 128, 256, 512)`` with a 1024-channel bottleneck give ~31M
    parameters. ``forward`` returns per-pixel class logits of shape ``(N, classes, H, W)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        widths: tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        prev = in_channels
        for w in widths:
            self.encoders.append(DoubleConv(prev, w))
            prev = w
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(prev, prev * 2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = prev * 2
        for w in reversed(widths):
            self.upsamples.append(nn.ConvTranspose2d(prev, w, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(w * 2, w))
            prev = w
        self.head = nn.Conv2d(prev, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips), strict=True):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # odd input sizes
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear")
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel class index, shape ``(N, H, W)``."""
        return self.forward(x).argmax(dim=1)
