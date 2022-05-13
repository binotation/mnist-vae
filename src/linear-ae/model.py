import numpy as np
import torch.nn as nn

class LinearAE(nn.Module):
    '''Autoencoder with only fully-connected layers. Latent dim=64.'''
    def __init__(self, data_dim):
        '''data_dim: (H, W, channels)'''
        self.data_dim = data_dim
        self._flat_data_dim = np.prod(data_dim)
        super(LinearAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_data_dim, 512),
            nn.Linear(512, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.Linear(512, self._flat_data_dim),
            nn.Sigmoid()
            # Reshape after
        )

    def forward(self, x):
        '''x: (batch, channels, H, W)
        Returns: (batch, channels, H, W)
        '''
        z = self.encoder(x)

        x_prime = self.decoder(z).reshape(-1, self.data_dim[2], self.data_dim[0], self.data_dim[1])
        return x_prime
