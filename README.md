# 3D U-Net for Brain Tumor Segmentation (BraTS)

This repository contains a PyTorch implementation of a 3D U-Net model designed for medical image segmentation, specifically targeting the BraTS (Brain Tumor Segmentation) dataset.

## Repository Structure

- `app.py`: Main application/API script.
- `dataset.py`: Dataset loading, augmentation, and processing pipeline.
- `model.py`: 3D U-Net architecture definition.
- `train.py`: Model training and validation loops.
- `loss.py`: Custom loss functions optimized for segmentation (e.g., Dice Loss).
- `metrics.py`: Evaluation metrics for assessing model performance.
- `preprocess.py`: Scripts for data preprocessing and normalization.
- `visualize.py`: Utilities for visualizing medical volumes and segmentation masks.
- `volumetry.py`: Tools for volumetric analysis and calculation.
- `eda.py`: Exploratory Data Analysis script.
- `main.py`: Entry point script.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd 3D_UNET
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   *(Ensure you have a `requirements.txt` file created with `pip freeze > requirements.txt`)*
   ```bash
   pip install -r requirements.txt
   ```

## Checkpoints and Data

Model checkpoints are stored in the `checkpoints/` directory, which is excluded from version control. Ensure you place your datasets in the appropriate data directories according to the `dataset.py` configuration.
