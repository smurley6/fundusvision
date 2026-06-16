# FundusVision: Automated Retinal Disease Detection

A comprehensive machine learning system for automated retinal disease detection from fundus images, developed as part of Georgia Tech's CS 7641 Machine Learning course.

## Overview

This project addresses a critical healthcare challenge: approximately 40 million people worldwide are blind, with 50-80% of cases preventable through early detection. Our system implements two complementary deep-learning approaches to analyze retinal fundus images: disease classification and vessel segmentation.

## Results

| Model | Task | Key Metrics |
|-------|------|-------------|
| **U-Net** | Vessel Segmentation | 97.6% pixel accuracy, 0.82 Dice score, 0.70 IoU |
| **ResNet34** | Binary Classification (Normal vs Abnormal) | Transfer learning on 6,392 per-eye images, patient-grouped split, threshold optimization |

## Approaches

### 1. U-Net Vessel Segmentation
CNN-based encoder-decoder architecture with skip connections for precise vessel boundary detection.
- **Architecture:** 31M parameters
- **Dataset:** 800 fundus images with ground-truth vessel masks
- **Output:** Binary segmentation masks highlighting retinal vasculature

### 2. ResNet34 Deep Learning Classification
Transfer learning approach using pretrained ImageNet weights for binary disease detection (Normal vs Abnormal).
- **Architecture:** ResNet34 backbone + custom 3-layer classifier (21.4M parameters)
- **Dataset:** ODIR-5K — 6,392 unique fundus images, labeled **per eye** from the diagnostic keywords
- **Split:** Patient-grouped train/val/test (no patient or image crosses splits), class-stratified
- **Features:** Class-weighted loss, cosine annealing LR, threshold optimization, comprehensive evaluation metrics

## Project Structure

```
fundusvision/
├── Segmentation/
│   ├── main.ipynb          # U-Net training and evaluation
│   ├── README.md           # Model documentation
│   └── dataset/            # Training and test images with masks
├── Deep Learning/
│   ├── main.ipynb          # ResNet34 classification pipeline
│   └── README.md           # Detailed model documentation
└── data/
    └── full_df.csv         # ODIR-5K metadata (images download from Kaggle)
```

> **Note:** The fundus images are **not** stored in this repo. The notebooks
> download the ODIR-5K dataset from Kaggle at runtime (see [Datasets](#datasets)).

## Datasets

- **[ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)** - 5,000+ fundus scans with disease labels and clinical notes
- **[Fundus Vessel Segmentation](https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation)** - 800 samples with ground-truth vessel masks

## Tech Stack

**Frameworks:** PyTorch, scikit-learn, OpenCV
**Models:** ResNet34, U-Net
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
