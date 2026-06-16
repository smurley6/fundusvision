# U-Net Vessel Segmentation Model

## Overview

This module implements a deep learning pipeline using a U-Net convolutional encoder–decoder for **retinal vessel segmentation on fundus images** — a per-pixel binary task that labels each pixel as **background** or **vessel**. The model is trained from scratch with ImageNet-normalized inputs and evaluated with pixel accuracy, IoU (Jaccard), and Dice score on a held-out test set.

## Model Architecture

**Pipeline:** Raw Image + Ground-Truth Mask → Preprocessing → U-Net Encoder → Bottleneck → U-Net Decoder (with skip connections) → Per-Pixel 2-Class Logits

**Backbone:** U-Net — a 4-level encoder/decoder built from scratch (no pretrained weights)

Each **conv block** is two 3×3 convolutions, each followed by BatchNorm and ReLU:

```
Conv2d(3×3, padding=1) → BatchNorm2d → ReLU
                       → Conv2d(3×3, padding=1) → BatchNorm2d → ReLU
```

- **Encoder:** 4 conv blocks (3→64→128→256→512 channels), each followed by 2×2 MaxPool, progressively halving spatial resolution.
- **Bottleneck:** one conv block (512→1024 channels) at the lowest resolution.
- **Decoder:** 4 stages, each a `ConvTranspose2d` (2×2, stride 2) upsampling step followed by concatenation of the matching encoder feature map (**skip connection**) and a conv block (1024→512→256→128→64 channels).
- **Output head:** a 1×1 convolution mapping the final 64 channels to **2 per-pixel class logits**.

## Dataset

- **Source:** A Kaggle fundus vessel-segmentation dataset with ground-truth vessel masks — downloaded manually from Kaggle (automatic download is disabled in the notebook) and placed in `dataset/`.
- **Layout:**
  ```
  Segmentation/dataset/
    train/
      Original/      *.png   (fundus images)
      Ground truth/  *.png   (vessel masks, same filenames)
    test/
      Original/      *.png
      Ground truth/  *.png
  ```
- **Sizes:** Train 480 · Validation 120 · Test 200
- **Split:** ~600 training images are split **80/20** into train/validation via `random_split` with a fixed seed (`manual_seed(42)`); the 200 test images are kept separate.
- **Masks:** loaded grayscale, resized to 224×224, and **binarized at pixel > 127** (1 = vessel, 0 = background).

## File Description

`main.ipynb` — Main Jupyter notebook containing:
  - Dataset path setup and a manual-download note (Kaggle auto-download is disabled)
  - A custom PyTorch `SegmentationDataset` pairing each image with its same-named mask
  - Device selection (CUDA → MPS → CPU) and a single shared image transform (resize + ImageNet normalization)
  - 80/20 train/validation split and DataLoader construction (worker parallelism, CUDA-gated `pin_memory`)
  - Sample visualization of images and masks
  - U-Net model definition (`SegmentationModel`) with conv blocks and skip connections
  - Training loop with a cosine LR schedule and best-checkpoint saving, reporting per-epoch train/validation loss and pixel accuracy
  - Test-set evaluation (pixel accuracy, IoU, Dice)

## Model Configuration

### Data Preprocessing
- **Image Size:** 224×224
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Color Space:** RGB (BGR→RGB conversion from OpenCV)
- **Masks:** resized to 224×224 and binarized at pixel > 127

### U-Net Architecture
- **Input:** 224×224×3 RGB images
- **Output:** 224×224×2 per-pixel class logits (background vs vessel)
- **Encoder/Decoder depth:** 4 levels with skip connections
- **Conv blocks:** double 3×3 Conv → BatchNorm → ReLU
- **Upsampling:** `ConvTranspose2d` (2×2, stride 2)
- **Total Parameters:** 31,043,586

### Training Configuration
- **Optimizer:** Adam (lr=1e-4)
- **Loss Function:** CrossEntropyLoss (per-pixel, 2 classes)
- **LR Scheduler:** CosineAnnealingLR (T_max = number of epochs), stepped once per epoch
- **Batch Size:** 8
- **Epochs:** 15
- **Checkpointing:** best model (lowest validation loss) saved as a pure `state_dict` to `best_segmentation_model.pth`
- **Device:** CUDA → MPS (Apple Silicon) → CPU (auto-selected)
- **DataLoaders:** `num_workers=2`; `pin_memory` enabled only on CUDA

## Performance Metrics

Recorded test-set results from a reference run:

| Metric                 | Value  |
| ---------------------- | ------ |
| Pixel Accuracy         | 0.9758 |
| IoU (Jaccard, vessel)  | 0.7010 |
| Dice                   | 0.8242 |

> **Note on pixel accuracy:** Vessels occupy only a small fraction of each image, so a model can score very high pixel accuracy (≈ 0.976) by predicting "background" almost everywhere. For thin-vessel masks, **IoU and Dice are the meaningful metrics** because they directly measure overlap with the (sparse) vessel class.

> **Note on numbers:** The values above are from the original 10-epoch reference run. With the cosine LR schedule and 15 epochs now in the notebook, re-run training to obtain current-configuration metrics.

## Key Implementation Details

1. **Standard U-Net** — 4-level encoder/decoder with skip connections that fuse high-resolution encoder features into the decoder, preserving fine vessel detail.
2. **BatchNorm + ReLU conv blocks** — every conv block uses two 3×3 convolutions with BatchNorm and ReLU for stable training from scratch.
3. **Device selection** — CUDA → MPS (Apple Silicon) → CPU, chosen automatically.
4. **Cosine LR schedule** — `CosineAnnealingLR` smoothly decays the learning rate over training.
5. **Best-checkpoint saving** — the lowest-validation-loss model is saved as a pure `state_dict` (`best_segmentation_model.pth`) rather than relying on last-epoch weights.
6. **Mask binarization** — ground-truth masks are thresholded at pixel > 127 to produce clean binary {background, vessel} labels.
7. **Seeded split** — `random_split` with `manual_seed(42)` for a reproducible train/validation partition.
8. **Class-imbalance-aware evaluation** — alongside pixel accuracy, the test cell reports IoU (`sklearn.metrics.jaccard_score`, `average='binary'`) and a manually computed Dice coefficient.

## Running the Notebook

1. **Google Colab:** Upload to Colab and run with a GPU runtime.
2. **Local run (Apple-Silicon MPS / CPU):** The notebook auto-selects the MPS backend on Apple Silicon (or CPU as a fallback). GPU is recommended — each epoch takes ~2–3 minutes on the reference run.
3. **Dataset:** Automatic Kaggle download is disabled. **Download the dataset from Kaggle manually** and place it under `Segmentation/dataset/` following the `train/{Original, Ground truth}` and `test/{Original, Ground truth}` layout above.
4. **Execution:** Run cells sequentially — build the dataset/loaders, define the U-Net, train for 15 epochs (best checkpoint saved automatically), then evaluate on the test set.

## Required Dependencies

- numpy, pandas
- matplotlib
- opencv-python (cv2)
- torch, torchvision
- scikit-learn
- tqdm

## Future Improvements

1. **Segmentation-aware losses** — Dice loss, Tversky loss, or a boundary loss to directly optimize overlap on thin vessels and counter class imbalance.
2. **Data augmentation** — flips, rotations, elastic/affine warps, and color jitter to improve generalization on a small dataset.
3. **More epochs / early stopping** — train longer with the schedule, stopping on validation plateau (loss was still decreasing at epoch 10 in the original run).
4. **Best checkpoint by IoU/Dice** — select the saved model on validation overlap rather than loss.
5. **Test-time augmentation (TTA)** — average predictions over flips/rotations.
6. **Attention U-Net** — attention gates on skip connections to focus on vessel regions.

## References

- Dataset: Kaggle fundus vessel-segmentation dataset (with ground-truth masks)
- Architecture: U-Net (Ronneberger et al., 2015)
- Framework: PyTorch 2.x
