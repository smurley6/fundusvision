"""Grad-CAM (Selvaraju et al., 2017) for the fundus classifier."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from fundusvision.data.odir import CLASS_NAMES, FundusDataset, build_transforms
from fundusvision.device import get_device
from fundusvision.evaluation.classifier import load_run
from fundusvision.evaluation.plots import COLORS, plot_image_grid


class GradCAM:
    """Class-activation maps from the gradients of the logit w.r.t. ``target_layer``."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(self, _module: nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cams in [0, 1] of shape (N, H, W), P(Abnormal) of shape (N,))``."""
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits.sum().backward()  # samples are independent, so this gives per-sample grads
        assert self._activations is not None and self._gradients is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo + 1e-8)).view_as(cam)
        return cam.float().cpu().numpy(), torch.sigmoid(logits.detach()).float().cpu().numpy()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def gradcam_overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET heat-map of ``cam`` over an RGB uint8 image (resized to the image)."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    if heat.shape[:2] != image_rgb.shape[:2]:
        heat = cv2.resize(heat, (image_rgb.shape[1], image_rgb.shape[0]))
    return cv2.addWeighted(image_rgb, 1 - alpha, heat, alpha, 0)


def make_gradcam_figure(
    run_dir: str | Path,
    out_path: str | Path,
    num_examples: int = 8,
    device: str | None = None,
    seed: int = 0,
) -> Path:
    """Grad-CAM overlays for correctly classified test eyes (half Normal, half Abnormal)."""
    dev = get_device(device)
    model, cfg, split = load_run(run_dir, dev)
    test = split[split["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    picks = pd.concat(
        [
            test[test["label"] == c].sample(
                n=min(num_examples // 2, (test["label"] == c).sum()),
                random_state=int(rng.integers(1 << 30)),
            )
            for c in (1, 0)
        ]
    ).reset_index(drop=True)
    ds = FundusDataset(picks, transform=build_transforms(cfg["image_size"], augment=False))
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)
    cam_fn = GradCAM(model, model.feature_layer)
    cams, probs = cam_fn(x)
    cam_fn.remove()

    images, titles, colors = [], [], []
    for i in range(len(ds)):
        img = cv2.resize(ds.image(i), (cfg["image_size"], cfg["image_size"]))
        images.append(gradcam_overlay(img, cams[i]))
        label = int(picks.loc[i, "label"])
        titles.append(f"true {CLASS_NAMES[label]} | P(abn) {probs[i]:.2f}")
        colors.append(COLORS["abnormal"] if label else COLORS["normal"])
    return plot_image_grid(
        images,
        titles,
        out_path,
        ncols=4,
        suptitle="Grad-CAM (layer4) on test eyes",
        title_colors=colors,
    )
