# Training Module

This module contains the training script for the undercomplete autoencoder used in the ALHO project.

## Files

- `train.py` - Main training script for the autoencoder model

## Overview

The `train.py` script implements a complete training pipeline for an undercomplete autoencoder that performs dimensionality reduction on URL feature data. The autoencoder compresses high-dimensional URL features into a lower-dimensional representation while attempting to preserve the most important information.

## Usage

```bash
python3 train.py <path_to_csv>
```

### Arguments

- `path_to_csv` - Path to the CSV file containing the URL feature data

### Example

```bash
python3 src/training/train.py dataset/organized/train.csv
```

## Script Functionality

### 1. Data Setup (`setup()`)

- Reads the input CSV file using pandas
- Removes the 'url' column (assuming it's not needed for training)
- Drops the first remaining column (likely the label column)
- Returns the feature matrix `X`

### 2. Model Training (`train()`)

- **Data Preprocessing**: Normalizes the input data using `StandardScaler` from scikit-learn
- **Model Architecture**: Creates an `UndercompleteAE` autoencoder with:
  - Input size: Number of features in the dataset
  - Encoding dimension: n (reduces to n-dimensional representation, set as 3 for testing purposes)
- **Training Configuration**:
  - Loss function: Mean Squared Error (MSE)
  - Optimizer: Adam with learning rate 0.003 (to be tuned)
  - Number of epochs: 20 (to be tuned)
  - Random seed: 42 (set only for reproducibility)
- **Training Loop**: Performs forward pass, loss calculation, backpropagation, and parameter updates
- **Output**: Returns the encoded (compressed) representation of the input data

### 3. Data Export (`export()`)

- Saves the encoded data to a CSV file
- Default filename: `encodedData.csv`
- Exports without index column

## Model Architecture

The script uses the `UndercompleteAE` class from `models.autoencoder`, which implements:

- **Encoder**: Compresses input features to a lower-dimensional latent space
- **Decoder**: Reconstructs the original input from the latent representation
- **Undercomplete**: The latent dimension is smaller than the input dimension, forcing the model to learn a compressed representation

## Dependencies

- `pandas` - Data manipulation and CSV handling
- `torch` - PyTorch deep learning framework
- `scikit-learn` - Data preprocessing (StandardScaler)
- `sys` - Command line argument handling

## Output

The script generates:
1. Training progress output showing loss for each epoch
2. A CSV file (`encodedData.csv`) containing the compressed 3-dimensional representation of the input data

## Notes

- The script is designed for URL feature data where dimensionality reduction is desired
- The encoding dimension is hardcoded to 3, which may need adjustment based on your specific use case
- Data normalization is applied to ensure stable training
- The model uses a fixed random seed for reproducible results
