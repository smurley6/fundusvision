# fundusvision
ML Project Repo - Automated Retinal Disease Detection


## Project Structure

### Segmentation Model

```
Segmentation/
│
├── dataset/           E
│   ├── train/                        # Training and validation data
│   │   ├── Original/
│   │   └── Ground truth/
│   └── test/                         # Test data
│       ├── Original/
│       └── Ground truth/
│
├── main.ipynb                        # Main Jupyter notebook with complete pipeline
└── best_model.pth                    # Saved best model weights (generated during training)

```

#### Segmentation File Descriptions

`main.ipynb` - Main Jupyter notebook containing:
  - Data loading and preprocessing
  - Model architecture definition (use U-Net as reference)
  - Training loop with validation
  - Model evaluation and visualization
  - Performance metrics (IoU, accuracy)

### Random Forest Classification Model

```
Random Forest/
│
├── main.ipynb                        # Main Jupyter notebook with complete pipeline
└── README.md                         # Model documentation and specifications

```

#### Random Forest File Descriptions

`main.ipynb` - Main Jupyter notebook containing:
  - Data loading from ODIR-5K dataset (2,492 validation images)
  - Image preprocessing (128×128 grayscale, augmentation)
  - PCA dimensionality reduction (200 components)
  - Random Forest classifier training (400 trees)
  - Multi-label evaluation for 8 disease categories
  - Performance analysis and improvement recommendations

`README.md` - Comprehensive documentation including:
  - Model architecture and pipeline overview
  - Dataset description and label distribution
  - PCA and Random Forest hyperparameters
  - Performance metrics and results analysis
  - Future improvement recommendations

### Deep Learning Classification Model

```
Deep Learning/
│
├── main.ipynb                        # Main Jupyter notebook with complete pipeline
└── README.md                         # Model documentation and specifications

```

#### Deep Learning File Descriptions

`main.ipynb` - Main Jupyter notebook containing:
  - Kaggle dataset download and authentication
  - Data loading from ODIR-5K dataset (12,460 total images)
  - Image preprocessing (224×224, ImageNet normalization)
  - Separate data augmentation for train/validation sets
  - ResNet34 model with custom classifier (21.4M parameters)
  - Training with AdamW optimizer and learning rate scheduling
  - Comprehensive evaluation (15+ metrics including ROC AUC, PR AUC)
  - Threshold optimization and error analysis

`README.md` - Comprehensive documentation including:
  - Model architecture (ResNet34 backbone + custom classifier)
  - Dataset split and class distribution
  - Data augmentation and preprocessing pipeline
  - Training configuration and hyperparameters
  - Performance metrics and evaluation results
  - Key improvements over baseline models
  - Future enhancement recommendations
