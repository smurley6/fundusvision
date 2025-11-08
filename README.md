# fundusvision
ML Project Repo


## Project Structure of Segmentation model

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

##  File Descriptions

`main.ipynb` - Main Jupyter notebook containing:
  - Data loading and preprocessing
  - Model architecture definition (use U-Net as reference)
  - Training loop with validation
  - Model evaluation and visualization
  - Performance metrics (IoU, accuracy)
