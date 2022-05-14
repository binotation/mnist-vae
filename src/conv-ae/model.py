import numpy as np
import torch.nn as nn
from helpers import get_convs_out_dim

class ConvAE(nn.Module):
    '''Autoencoder with conv2d and convtranspose2d layers. Latent dim=64. The convolutional
    layers are loosely based on resnet18.
    '''
    def __init__(self, data_dim):
        '''data_dim: (H, W, channels)'''
        super(ConvAE, self).__init__()

        self.data_dim = data_dim
        # (h, w, channels)
        self._conv_out_dim = get_convs_out_dim(data_dim,\
            (32, 32, 64, 128, 256, 256),\
            (6,  2,  3,  3,   3,   3),\
            (2,  2,  1,  1,   1,   3),\
            (2,  1,  0,  0,   0,   0))

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=data_dim[2], out_channels=32, kernel_size=6, stride=2, padding=2), # 16
            nn.MaxPool2d(kernel_size=2, padding=1), # 9

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0), # 7
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0), # 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0), # 3

            nn.AvgPool2d(kernel_size=3),
        )

        self.encoder2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._conv_out_dim[2], 64)
        )

        self.encoder = nn.Sequential(
            self.encoder1,
            self.encoder2
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(64, self._conv_out_dim[2])
            # Reshape after
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=3, padding=0),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=0),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=0),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=0),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=data_dim[2], kernel_size=6, stride=2, padding=2),
            nn.Sigmoid(), # Sigmoid here works better than Tanh
        )

    def forward(self, x):
        '''x: (batch, channels, H, W)
        Returns: (batch, channels, H, W)
        '''
        z = self.encoder(x)
        h, w, channels = self._conv_out_dim
        z = self.decoder1(z).reshape(-1, channels, h, w)

        x_prime = self.decoder2(z)
        return x_prime
