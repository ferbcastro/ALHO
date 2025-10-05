import torch
import torch.nn as nn

""" TODO : Make generic class that takes into a configuration
object and generates a feed forward net, then the encoder
and decoder can be instaces with diffents objects. Also
will make fine tunning easier
"""

class Encoder(nn.Module):
    def __init__(self, input_dim=9860, latent_dim=1024):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(input_dim, 4930),
                nn.ReLU(), 
                nn.Linear(4930, 2450),
                nn.ReLU(), 
                nn.Linear(2450, latent_dim)
                )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, output_dim=9860, latent_dim=1024):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(latent_dim, 2450),
                nn.ReLU(),
                nn.Linear(2450, 4930),
                nn.ReLU(),
                nn.Linear(4930, output_dim),
                nn.Sigmoid() 
                )

    def forward(self, x):
        return self.layers(x)

class UndercompleteAE(nn.Module):
    def __init__(self, input_dim=9860, latent_dim=1024):
        super().__init__()         
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded 
