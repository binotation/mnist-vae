import numpy as np
import torch.nn as nn
from helpers import get_convs_out_dim

class ConvAE(nn.Module):
    '''Autoencoder with conv2d and convtranspose2d layers. Latent dim=64. The convolutional
    layers are based on lenet5.
    '''
    def __init__(self, data_dim):
        '''data_dim: (H, W, channels)'''
        super(ConvAE, self).__init__()

        self.data_dim = data_dim
        # (h, w, channels)
        self._conv_out_dim = get_convs_out_dim(data_dim, (6, 6, 16, 16), (5, 2, 5, 2), (1, 2, 1, 2), (0,)*4)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=data_dim[2], out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(np.prod(self._conv_out_dim), 64),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(64, np.prod(self._conv_out_dim))
            # Reshape after
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5),
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=6, out_channels=data_dim[2], kernel_size=5),
            nn.Sigmoid(), # Sigmoid here works better than Tanh
        )

    def forward(self, x):
        '''x: (batch, channels, H, W)
        Returns: (batch, channels, H, W)
        '''
        z = self.encoder(x)
        conv_out_dim = self._conv_out_dim
        z = self.decoder1(z).reshape(-1, conv_out_dim[2], conv_out_dim[0], conv_out_dim[1])

        x_prime = self.decoder2(z)
        return x_prime
