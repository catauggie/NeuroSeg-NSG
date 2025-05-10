# U-Net Model for Live/Dead Cell Segmentation on Trypan Blue-Stained Images

This repository contains code and configurations for training a U-Net model to segment live and dead cells on trypan blue–stained images, commonly used in biomedical image analysis. The model focuses on accurately identifying and distinguishing cells based on their staining properties in bright-field images, a crucial step in many biological experiments.

## Contents:

- **get_metrics.py**: A Python script for calculating segmentation metrics, including Dice coefficient, precision, recall, and others, to evaluate model performance.
- **torch_config.py**: Contains configuration settings for training the U-Net model in PyTorch, including parameters for the optimizer, learning rate, augmentation, and more.
- **torch_image_to_tensor.py**: Utility script to convert raw image data into tensors suitable for input into the U-Net model.
- **torchjnetrics.py**: Script to compute additional performance metrics during training, useful for model evaluation and hyperparameter tuning.
- **torch_training.py**: Contains the training loop for the U-Net model, including data loading, model evaluation, and checkpointing.
- **torch_unet_universal.py**: The core U-Net model implementation in PyTorch. It can be adapted to various biomedical segmentation tasks by modifying the configuration and training parameters.

## Description:

This U-Net model is specifically designed to segment live and dead cells in trypan blue-stained images, a popular staining method used in cell biology to assess cell viability. The training pipeline leverages data augmentation techniques and early stopping to improve model performance and prevent overfitting. The provided scripts also include metric calculation to help assess the quality of segmentation results.

## Usage:

1. **Install Dependencies**: Ensure that all necessary dependencies, such as `torch` and `torchvision`, are installed in your Python environment.
   
   ```bash
   pip install torch torchvision
