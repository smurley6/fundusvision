# FundusVision: Automated Retinal Disease Detection

A comprehensive machine learning system for automated retinal disease detection from fundus images, developed as part of Georgia Tech's CS 7641 Machine Learning course.

## Overview

This project addresses a critical healthcare challenge: approximately 40 million people worldwide are blind, with 50-80% of cases preventable through early detection. Our system implements three complementary approaches to analyze retinal fundus images for disease classification and vessel segmentation.

## Results

| Model | Task | Key Metrics |
|-------|------|-------------|
| **U-Net** | Vessel Segmentation | 97.6% pixel accuracy, 0.82 Dice score, 0.70 IoU |
| **ResNet34** | Binary Classification | Transfer learning on 12,460 images with threshold optimization |
| **PCA + Random Forest** | Multi-label Classification | 8 disease categories, 99.6% dimensionality reduction |

## Approaches

### 1. U-Net Vessel Segmentation
CNN-based encoder-decoder architecture with skip connections for precise vessel boundary detection.
- **Architecture:** 31M parameters
- **Dataset:** 800 fundus images with ground-truth vessel masks
- **Output:** Binary segmentation masks highlighting retinal vasculature

### 2. ResNet34 Deep Learning Classification
Transfer learning approach using pretrained ImageNet weights for binary disease detection (Normal vs Abnormal).
- **Architecture:** ResNet34 backbone + custom 3-layer classifier (21.4M parameters)
- **Dataset:** ODIR-5K (12,460 fundus images)
- **Features:** Class-weighted loss, cosine annealing LR, comprehensive evaluation metrics

### 3. PCA + Random Forest Baseline
Traditional ML pipeline establishing performance benchmarks for comparison.
- **Pipeline:** 128×128 grayscale → PCA (200 components) → Random Forest (400 trees)
- **Dataset:** ODIR-5K validation set (2,492 images)
- **Labels:** Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other

## Project Structure

```
fundusvision/
├── Segmentation/
│   ├── main.ipynb          # U-Net training and evaluation
│   └── dataset/            # Training and test images with masks
├── Deep Learning/
│   ├── main.ipynb          # ResNet34 classification pipeline
│   └── README.md           # Detailed model documentation
├── Random Forest/
│   ├── main.ipynb          # PCA + RF baseline model
│   └── README.md           # Model specifications
└── data/                   # Shared dataset utilities
```

## Datasets

- **[ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)** - 5,000+ fundus scans with disease labels and clinical notes
- **[Fundus Vessel Segmentation](https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation)** - 800 samples with ground-truth vessel masks

## Tech Stack

**Frameworks:** PyTorch, scikit-learn, OpenCV
**Models:** ResNet34, U-Net, Random Forest
**Tools:** Jupyter, Google Colab, Kaggle API

## Team

- **Steven Murley** - Project Manager
- **Sohum Joshi** - Infrastructure & Technical Lead
- **Adam Bakr** - Data Analyst & Research Specialist
- **Yuhan Wei** - Research & Evaluation Specialist
- **Connor Smith** - Research Writer & Documentation Lead

## Links

- [Project Website](https://mlproject-blue.vercel.app) - Detailed methodology and results
- [Course](https://omscs.gatech.edu/cs-7641-machine-learning) - CS 7641 Machine Learning

---

*Georgia Institute of Technology | CS 7641 Machine Learning | Fall 2025*
