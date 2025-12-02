# Deep Learning Classification Model

## Overview

This module implements an improved deep learning pipeline using ResNet34 with comprehensive data augmentation, learning rate scheduling, and advanced evaluation metrics for binary classification of fundus images (Normal vs Abnormal) on the ODIR-5K dataset.

## Model Architecture

**Pipeline:** Raw Images → Preprocessing → Data Augmentation → ResNet34 Backbone → Custom Classifier → Binary Predictions

**Backbone:** ResNet34 (pretrained on ImageNet)
**Classifier:** 3-layer fully connected network with BatchNorm and Dropout

## Dataset

- **Source:** ODIR-5K (Ocular Disease Intelligent Recognition)
- **Total Images:** 12,460 fundus images (left and right eyes combined)
- **Training Set:** 8,722 images (70%)
- **Validation Set:** 1,869 images (15%)
- **Test Set:** 1,869 images (15%)
- **Task:** Binary classification (Normal vs Abnormal)
- **Class Distribution:**
  - Normal: 2,101 patients (33.7%)
  - Abnormal: 4,291 patients (66.3%)

## Project Structure

```
Deep Learning/
│
├── main.ipynb                        # Main Jupyter notebook with complete pipeline
└── README.md                         # This file
```

## File Description

`main.ipynb` - Main Jupyter notebook containing:
  - Kaggle dataset download and authentication
  - Data loading and image-label pair creation
  - Separate transforms for training (with augmentation) and validation (without augmentation)
  - Custom PyTorch Dataset and DataLoader implementation
  - ResNet34 model architecture with custom classifier head
  - Training loop with learning rate scheduling and early stopping
  - Comprehensive evaluation with 15+ metrics
  - Threshold optimization using F1 score and Youden's Index
  - Error analysis and misclassification visualization
  - Model checkpointing and result export

## Model Configuration

### Data Preprocessing
- **Image Size:** 224×224
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Color Space:** RGB (BGR→RGB conversion from OpenCV)

### Data Augmentation (Training Only)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.2)
- Random rotation (±15°)
- Color jitter (brightness: 0.7-1.3, contrast/saturation: ±0.2)
- Random affine (translation: ±5%, shear: 5°)
- Gaussian blur (kernel=3, p=0.3)

### ResNet34 Architecture
- **Backbone:** ResNet34 pretrained on ImageNet1K
- **Input:** 224×224×3 RGB images
- **Feature Dimensions:** 512 (after global average pooling)
- **Total Parameters:** 21,449,922
- **Trainable Parameters:** 21,449,922

### Custom Classifier Head
```
512 features → Linear(512, 256) → BatchNorm1d → ReLU → Dropout(0.5)
            → Linear(256, 128) → BatchNorm1d → ReLU → Dropout(0.25)
            → Linear(128, 2) → Sigmoid
```

### Training Configuration
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss Function:** Binary Cross-Entropy Loss with class weights
  - Normal weight: 1.504
  - Abnormal weight: 0.749
- **Learning Rate Scheduler:** CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-6)
- **Batch Size:** 16
- **Max Epochs:** 50
- **Early Stopping:** Patience of 10 epochs
- **Device:** CUDA (GPU acceleration)

## Performance Metrics

The model achieves comprehensive evaluation across multiple dimensions:

### Classification Metrics (Test Set, Optimal Threshold)
- **Accuracy:** Reported in notebook
- **Balanced Accuracy:** Accounts for class imbalance
- **Precision (PPV):** Positive Predictive Value
- **Recall (Sensitivity):** True Positive Rate
- **Specificity:** True Negative Rate
- **F1 Score:** Harmonic mean of precision and recall
- **NPV:** Negative Predictive Value

### Discrimination Metrics
- **ROC AUC:** Area Under the Receiver Operating Characteristic Curve
- **PR AUC:** Area Under the Precision-Recall Curve
- **Optimal Threshold:** Determined via F1 score and Youden's Index optimization

### Agreement Metrics
- **Matthews Correlation Coefficient (MCC):** Overall quality measure
- **Cohen's Kappa:** Agreement beyond chance
- **Youden's Index:** Sensitivity + Specificity - 1

### Confusion Matrix Components
- True Positives (TP), True Negatives (TN)
- False Positives (FP), False Negatives (FN)

## Key Improvements Over Baseline

This improved deep learning model addresses limitations of traditional ML approaches:

1. **Separate Transforms:** Training uses augmentation; validation/test do not (prevents data leakage)
2. **Better Architecture:** ResNet34 instead of ResNet18 with enhanced classifier head
3. **Learning Rate Scheduling:** Prevents training plateaus and enables better convergence
4. **Class Weighting:** Handles imbalanced dataset (66.3% abnormal vs 33.7% normal)
5. **Comprehensive Evaluation:** 15+ metrics with advanced visualizations
6. **Threshold Optimization:** Finds optimal decision boundary for classification
7. **Error Analysis:** Visualizes misclassifications (false positives and false negatives)
8. **Real-time Monitoring:** Training metrics plotted live during training

## Visualizations

The notebook generates comprehensive visualizations including:
- Training history (loss, accuracy, F1, learning rate)
- Class distribution analysis
- Sample fundus images with labels
- Confusion matrix (counts and percentages)
- ROC curve and Precision-Recall curve
- Prediction probability distributions
- Metrics comparison bar charts
- Threshold optimization curves
- Misclassification examples (false positives and false negatives)

## Running the Notebook

1. **Google Colab:** Upload to Colab and run with GPU runtime
2. **Kaggle Authentication:** Upload `kaggle.json` API key when prompted
3. **Dataset Download:** Automatic via Kaggle API (ODIR-5K dataset)
4. **Dependencies:** Automatically installed via `!pip install` command
5. **Execution:** Run cells sequentially to reproduce results
6. **Output:** Model checkpoints (`best_model.pth`, `final_model_complete.pth`) and evaluation plots

## Required Dependencies

- pandas, numpy
- matplotlib, seaborn
- opencv-python (cv2)
- torch, torchvision
- scikit-learn
- tqdm
- kaggle

## Clinical Relevance

This binary classification model serves as a screening tool for fundus images:
- **High Sensitivity:** Minimizes false negatives (missed diseases)
- **High Specificity:** Minimizes false positives (unnecessary referrals)
- **Optimal Threshold:** Balances sensitivity and specificity for clinical use
- **Comprehensive Metrics:** Provides clinically relevant evaluation (PPV, NPV, etc.)

## Future Improvements

1. **Multi-label Classification:** Predict specific diseases (Diabetes, Glaucoma, AMD, etc.)
2. **Ensemble Methods:** Combine multiple models (ResNet34, ResNet50, EfficientNet)
3. **Attention Mechanisms:** Add spatial attention to focus on lesion regions
4. **External Validation:** Test on other datasets (Messidor, EyePACS, etc.)
5. **Explainability:** Implement Grad-CAM for visualization of decision regions
6. **Test-Time Augmentation:** Average predictions over multiple augmented versions
7. **Advanced Architectures:** Experiment with Vision Transformers (ViT) or Swin Transformers

## References

- Dataset: ODIR-5K (Ocular Disease Intelligent Recognition)
- Backbone: ResNet34 (He et al., 2015)
- Framework: PyTorch 2.9.0+
