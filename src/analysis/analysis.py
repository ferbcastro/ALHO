import torch
import torch.nn as nn

from models.autoencoder import UndercompleteAE 
from torch.utils.data import DataLoader, TensorDataset

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.BCELoss();
    total_loss = 0.0;
    num_batches = 0;

    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X[0].to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = UndercompleteAE(input_dim=22, latent_dim=2048).to(device)
    model.load_state_dict(torch.load("./UCAE_state", map_location=device))

    # aqui vai o path do arquivo teste
    test_path = ".////"

    X_test = torch.load(test_path)
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # avaliar modelo
    reconstruction_loss = evaluate(model, test_loader, device)
    print(f"Loss medio no conjunto de teste: {reconstruction_loss:.6f}")
