import torch.nn as nn

class LinearAE(nn.Module):
    '''Autoencoder with only fully-connected layers. Latent dim=64.'''
    def __init__(self):
        super(LinearAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.Linear(512, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.Linear(512, 784),
            nn.Sigmoid()
            # Reshape after
        )

    def forward(self, x):
        z = self.encoder(x)

        # Output dimensions (batch, H, W)
        x_prime = self.decoder(z).reshape(-1, 28, 28)
        return x_prime
