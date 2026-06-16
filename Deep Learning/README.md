# Deep Learning Classification Model

## Overview

This module implements a deep learning pipeline using ResNet34 (transfer learning) with data augmentation, learning-rate scheduling, threshold optimization, and comprehensive evaluation metrics for **binary classification of fundus images (Normal vs Abnormal)** on the ODIR-5K dataset.

## Model Architecture

**Pipeline:** Raw Images → Preprocessing → Data Augmentation → ResNet34 Backbone → Custom Classifier → Binary Prediction

**Backbone:** ResNet34 (pretrained on ImageNet)
**Classifier:** 3-layer fully connected network with BatchNorm and Dropout

## Dataset

- **Source:** ODIR-5K (Ocular Disease Intelligent Recognition) — downloaded from Kaggle at runtime
- **Unique images:** 6,392 fundus photographs (each physical image used exactly once)
- **Labels:** Derived **per eye** from that eye's diagnostic keyword — an eye is *Normal* only when its keyword is `normal fundus` (ignoring non-pathological capture artifacts such as `lens dust`); otherwise *Abnormal*
- **Split:** **Patient-grouped** and class-stratified via `StratifiedGroupKFold`, so no patient (and therefore no image) appears in more than one split:
  - Train: 4,560 images (≈ 2,398 patients, ≈ 55% abnormal)
  - Validation: 923 images (≈ 480 patients, ≈ 54% abnormal)
  - Test: 909 images (≈ 480 patients, ≈ 56% abnormal)

> **Why per-eye + patient-grouped?** The earlier pipeline looped over both eyes of
> every CSV row, duplicating each image ~1.95× (6,392 images → "12,460 pairs"), then
> split at the row level — placing the *same* image in both train and test. It also
> applied the patient-level Normal flag to both eyes. The current pipeline deduplicates,
> labels each eye independently, and groups the split by patient to eliminate leakage.

## File Description

`main.ipynb` — Main Jupyter notebook containing:
  - Kaggle dataset download and authentication
  - Per-eye label construction, deduplication, and patient-grouped split (with leakage assertions)
  - Separate transforms for training (augmented) and validation/test (not augmented)
  - Custom PyTorch Dataset and DataLoader implementation
  - ResNet34 model with custom classifier head
  - Training loop with LR scheduling, early stopping, and checkpointing
  - Comprehensive evaluation (15+ metrics) and threshold optimization
  - Error analysis and misclassification visualization

## Model Configuration

### Data Preprocessing
- **Image Size:** 224×224
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Color Space:** RGB (BGR→RGB conversion from OpenCV)

### Data Augmentation (Training Only)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.2)
- Random rotation (±15°)
- Color jitter (brightness 0.7–1.3, contrast/saturation ±0.2)
- Random affine (translation ±5%, shear 5°)
- Gaussian blur (kernel=3, p=0.3)

### ResNet34 Architecture
- **Backbone:** ResNet34 pretrained on ImageNet1K
- **Input:** 224×224×3 RGB images
- **Feature Dimensions:** 512 (after global average pooling)
- **Total Parameters:** 21,449,922

### Custom Classifier Head
```
512 features → Linear(512, 256) → BatchNorm1d → ReLU → Dropout(0.5)
            → Linear(256, 128) → BatchNorm1d → ReLU → Dropout(0.25)
            → Linear(128, 2) → Sigmoid
```

### Training Configuration
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss Function:** BCE loss with class weights computed from the **training split**
- **Learning Rate Scheduler:** CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-6)
- **Batch Size:** 16
- **Max Epochs:** 50, **Early Stopping:** patience of 10 epochs
- **Device:** CUDA → MPS (Apple Silicon) → CPU (auto-selected)

## Performance Metrics

The notebook reports a full battery of metrics on the test set at the optimal threshold:
classification (accuracy, balanced accuracy, precision/PPV, recall/sensitivity, specificity,
NPV, F1), discrimination (ROC AUC, PR AUC), agreement (MCC, Cohen's Kappa, Youden's Index),
and the confusion-matrix components.

> **Note on prior numbers:** Any classification metrics reported before the leakage fix
> (image-level split with duplicated images) were optimistically inflated. Re-run the
> notebook on the corrected patient-grouped split to obtain honest performance figures.

## Key Implementation Details

1. **No data leakage** — deduplicated images, per-eye labels, patient-grouped class-stratified split.
2. **Separate transforms** — augmentation on train only; validation/test untouched.
3. **Correct checkpointing** — the best model is saved as a pure `state_dict` and reloaded
   reliably (the original `torch.load` raised `UnpicklingError` under PyTorch ≥ 2.6's
   `weights_only=True` default and silently left evaluation on stale last-epoch weights).
4. **Learning-rate scheduling** — CosineAnnealingWarmRestarts.
5. **Threshold optimization** — optimal decision boundary via F1 score and Youden's Index.
6. **Comprehensive evaluation + error analysis** — 15+ metrics, ROC/PR curves, misclassification visualization.

## Running the Notebook

1. **Google Colab:** Upload to Colab and run with a GPU runtime.
2. **Kaggle Authentication:** Upload `kaggle.json` when prompted.
3. **Dataset Download:** Automatic via the Kaggle API (ODIR-5K).
4. **Local run (Apple Silicon / CPU):** The data paths auto-detect a local `data/` checkout;
   training uses MPS when available. (Images must be present locally — they are not stored in git.)
5. **Execution:** Run cells sequentially. Outputs: `best_model.pth` / `best_model_weights.pth` and evaluation plots.

## Required Dependencies

- pandas, numpy
- matplotlib, seaborn
- opencv-python (cv2)
- torch, torchvision
- scikit-learn
- tqdm
- kaggle

## Clinical Relevance

This binary classifier is a **screening** tool for fundus images:
- **High sensitivity:** minimizes false negatives (missed disease)
- **High specificity:** minimizes false positives (unnecessary referrals)
- **Optimal threshold:** balances sensitivity and specificity for clinical use

## Future Improvements

1. **Multi-label classification** — predict specific diseases (Diabetes, Glaucoma, AMD, etc.)
2. **Ensemble methods** — combine ResNet34, ResNet50, EfficientNet
3. **Attention mechanisms** — focus on lesion regions
4. **External validation** — Messidor, EyePACS, etc.
5. **Explainability** — Grad-CAM over decision regions
6. **Test-time augmentation**
7. **Advanced architectures** — Vision Transformers (ViT), Swin Transformers

## References

- Dataset: ODIR-5K (Ocular Disease Intelligent Recognition)
- Backbone: ResNet34 (He et al., 2015)
- Framework: PyTorch 2.x
