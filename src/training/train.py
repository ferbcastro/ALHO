''' Train an undercomplete autoencoder on the provided CSV data 
    and export the encoded data to a new CSV file.
'''

import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
from models.autoencoder import UndercompleteAE



if len(sys.argv) < 2:
    print("Usage: python3 train.py path_to_csv")
    sys.exit(1)

CSV_SOURCE = sys.argv[1]

def setup():
    """Sets up the environment by reading the CSV file and preparing the data."""

    df = pd.read_csv(CSV_SOURCE)

    df = df.drop('url', axis=1)

    X = df

    return X


def train(X: pd.DataFrame):
    """Trains the autoencoder on the provided data."""

    # Normalizing Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Converting to PyTorch tensor
    X_tensor = torch.FloatTensor(X_scaled)

    # Setting random seed for reproducibility
    torch.manual_seed(42)

    input_size = X.shape[1]  # Number of input features
    encoding_dim = 3  # Desired number of output dimensions
    model = UndercompleteAE(input_size, encoding_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0)

    # Training the autoencoder
    num_epochs = 20
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    encoded_data = model.encoder(X_tensor).detach().numpy()

    return encoded_data
    


def export(encoded_data, filename="encodedData.csv") -> None:
    """Exports the encoded data to a CSV file."""

    encoded_df = pd.DataFrame(encoded_data)
    encoded_df.to_csv(filename, index=False)
    print(f"Encoded data exported to {filename}")


def main():
    """Main function to train the autoencoder and export encoded data."""

    X = setup()

    encoded_data = train(X)

    export(encoded_data)

if __name__ == "__main__":
    main()
