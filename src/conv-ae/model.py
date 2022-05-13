import torch.nn as nn

class ConvAE(nn.Module):
    '''Autoencoder with conv2d and convtranspose2d layers. Latent dim=64. The convolutional
    layers are based on lenet5.
    '''
    def __init__(self):
        super(ConvAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(256, 64),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(64, 256)
            # Reshape after
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5),
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=5),
            nn.Sigmoid(), # Sigmoid here works better than Tanh
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder1(z).reshape(-1, 16, 4, 4)

        # Output dimensions (batch, H, W)
        x_prime = self.decoder2(z).squeeze(1)
        return x_prime
