''' Train an undercomplete autoencoder on the provided CSV data
    and export the encoded data to a new CSV file.
'''

import sys
import os

import pandas as pd

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
from models.autoencoder import UndercompleteAE

if len(sys.argv) < 2:
    print("Usage: python3 src/training/train.py path_to_csv [batch_size]")
    print("  path_to_csv: Path to the CSV file containing the data")
    print("  batch_size: Optional batch size for training (default: 32)")
    sys.exit(1)

CSV_SOURCE = sys.argv[1]
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 32

def setup():
    """Sets up the environment by reading the CSV file and preparing the data."""

    df = pd.read_csv(CSV_SOURCE)

    df = df.drop('url', axis=1)
    remaining_labels = df.pop('label')

    X = df

    return X, remaining_labels

def train(X: pd.DataFrame, batch_size: int = 32):
    """Trains the autoencoder on the provided data using batch processing."""

    # Converting to PyTorch tensor
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)

    # Create dataset and dataloader for batch processing
    dataloader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # Setting random seed for reproducibility
    torch.manual_seed(42)

    input_size = X.shape[1]  # Number of input features
    encoding_dim = 3  # Desired number of output dimensions
    model = UndercompleteAE(input_size, encoding_dim)

    # Loss function and optimizer
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.003, weight_decay=0)

    # Training the autoencoder
    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in dataloader:
            batch_X = batch_data[0]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # Generate encoded data for the entire dataset
    # model.eval()  # Set to evaluation mode
    # with torch.no_grad():
    #    encoded_data = model.encoder(X_tensor).detach().numpy()

    return model

def export_data(encoded_data, filename="encodedData.csv", labels: pd.DataFrame=None) -> None:
    """Exports the encoded data to a CSV file."""

    encoded_df = pd.DataFrame(encoded_data)

    if labels is not None:
        encoded_df = encoded_df.join(labels)

    encoded_df.to_csv(filename, index=False)

    print(f"Encoded data exported to {filename}")


def main():
    """Main function to train the autoencoder and export encoded data."""

    X, labels = setup()

    model = train(X, batch_size=BATCH_SIZE)

    # export_data(encoded_data, labels=labels)
    model.export()

if __name__ == "__main__":
    main()
