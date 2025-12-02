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
