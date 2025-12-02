# Random Forest Classification Model

## Overview

This module implements a baseline multi-label classification pipeline using PCA (Principal Component Analysis) for dimensionality reduction combined with Random Forest for detecting 8 eye diseases from fundus images on the ODIR-5K dataset.

## Model Architecture

**Pipeline:** Raw Images → Preprocessing → PCA (200 components) → Random Forest (400 trees) → Multi-label Predictions

## Dataset

- **Source:** ODIR-5K (Ocular Disease Intelligent Recognition)
- **Validation Set:** 2,492 fundus images
- **Task:** Multi-label binary classification
- **Labels:** 8 disease categories (Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other)

## Project Structure

```
Random Forest/
│
├── main.ipynb                        # Main Jupyter notebook with complete pipeline
└── README.md                         # This file
```

## File Description

`main.ipynb` - Main Jupyter notebook containing:
  - Data loading from ODIR-5K dataset
  - Image preprocessing (resize to 128×128, grayscale conversion, augmentation)
  - Feature extraction (flattening to 16,384-dimensional vectors)
  - PCA dimensionality reduction (200 components with whitening)
  - Random Forest classifier training (400 trees)
  - Model evaluation with per-label metrics
  - Visualization of PCA components, training progress, and confusion matrices
  - Performance analysis and recommendations for improvements

## Model Configuration

### PCA Parameters
- **Components:** 200
- **Whitening:** True
- **Random State:** 42
- **Input Dimensions:** 16,384 (128×128 grayscale images)

### Random Forest Parameters
- **Estimators:** 400 trees
- **Max Depth:** None (unlimited)
- **OOB Score:** True (out-of-bag error tracking)
- **Parallel Jobs:** -1 (utilize all CPU cores)

## Performance Metrics

The baseline model establishes benchmarks for:
- Per-label accuracy, precision, recall, and F1-score
- Confusion matrices for each disease category
- Class distribution analysis and imbalance ratios
- PCA explained variance analysis

## Key Findings

This baseline model demonstrates the limitations of traditional ML approaches for medical image classification:
- Strong performance on negative class prediction (high specificity)
- Limited positive disease detection (low sensitivity)
- Provides foundation for comparison with CNN-based approaches

## Future Improvements

Recommended next steps include:
1. Implement class balancing (SMOTE or class weights)
2. Transition to CNN architectures (ResNet, EfficientNet)
3. Apply transfer learning from medical imaging models
4. Preserve RGB color information (currently grayscale)
5. Increase image resolution (224×224 or higher)

## Running the Notebook

1. Ensure ODIR-5K dataset is available in the appropriate path
2. Install required dependencies: `pandas`, `numpy`, `scikit-learn`, `torch`, `torchvision`, `matplotlib`, `seaborn`
3. Open `main.ipynb` in Jupyter Notebook or Google Colab
4. Run cells sequentially to reproduce results

## References

See the comprehensive technical documentation in `eye_disease_detection_baseline.md` for detailed analysis, results, and recommendations.
