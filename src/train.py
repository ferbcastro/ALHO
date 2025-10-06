import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

from models.autoencoder import UndercompleteAE

df = pd.read_csv("./dataset/bigrams/train_set_bigrams.csv").head(6)
df = df.drop('url', axis=1)

X, Y = df.drop(df.columns[0], axis=1), df[df.columns[1:]]


# Normalizing Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converting to PyTorch tensor
X_tensor = torch.FloatTensor(X_scaled)

def main():
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

    # Encoding the data using the trained autoencoder
    encoded_data = model.encoder(X_tensor).detach().numpy()
    
    # Export as CSV
    encoded_df = pd.DataFrame(encoded_data)
    encoded_df.to_csv("encodedData.csv", index=False)



if __name__ == "__main__":
    main()