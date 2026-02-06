
# Retina Vessel Segmentation

A PyTorch implementation of U-Net for automated retinal blood vessel segmentation from fundus images using the DRIVE dataset.

## Overview

This project implements a deep learning pipeline for segmenting blood vessels in retinal fundus photographs. The model uses U-Net architecture with:

- **Encoder-Decoder** structure with skip connections
- **Dice Loss + BCE Loss** combination for handling class imbalance
- **Data augmentation** (elastic transform, rotation, flip) for robust training
- **CLAHE preprocessing** to enhance vessel contrast

## Project Structure

```
RetinaVesselSeg/
├── model.py           # U-Net architecture definition
├── dataset.py         # Dataset loader with augmentation
├── train.py           # Training script
├── predict.py         # Inference script
├── requirements.txt   # Dependencies
└── data/DRIVE/        # Dataset directory
    ├── training/
    ├── test/
    └── checkpoints/   # Saved models
```

## Requirements

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download the DRIVE dataset from [here](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) and organize as:

```
data/DRIVE/
├── training/images/
├── training/1st_manual/
├── test/images/
└── test/1st_manual/
```

## Usage

### Training

```bash
python src/train.py
```

**Hyperparameters:**
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 20
- Optimizer: Adam
- Loss: Dice + BCE

Monitor training with TensorBoard:

```bash
tensorboard --logdir=runs/experiment_1
```

### Inference

```bash
python predict.py
```

Modify `test_img_path` and `model_path` in `predict.py` as needed. Output shows:
1. Original image
2. Probability heatmap
3. Binary vessel mask (threshold = 0.5)

## Model Architecture

**Encoder:** 4 down-sampling blocks → Bottleneck
**Decoder:** 4 up-sampling blocks with skip connections
**Output:** Sigmoid activation for pixel-wise probability

## Key Features

- ✅ Dropout regularization in bottleneck layer
- ✅ Dynamic padding to handle odd-sized inputs
- ✅ Support for both ConvTranspose and bilinear upsampling
- ✅ TensorBoard logging for training visualization
- ✅ GPU/CPU automatic detection

## Results

The trained model achieves strong performance on DRIVE test set with proper preprocessing and augmentation strategies.
