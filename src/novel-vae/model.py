# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class VAE(nn.Module):
    '''Variational autoencoder. p_x is chosen as Bernoulli.'''
    def __init__(self, data_dim):
        super(VAE, self).__init__()
        self._latent_size = 2048
        self.inference_network = InferenceNetwork(data_dim, self._latent_size)
        self.generative_network = GenerativeNetwork(data_dim, self._latent_size)

    def forward(self, x):
        '''x: (batch, channels, H, W)
        Returns:
            - p_z (standard normal),
            - q_z (normal distribution used to calculate regularization term),
            - p_x_logits (p_x params, also interpreted as the generated data)
        '''
        batch_size = x.shape[0]

        # Retrieve q_z parameters and sample latent variable from q_z
        q_z_loc, q_z_scale = self.inference_network(x) # shape: (batch, latent)
        q_z = Normal(q_z_loc, q_z_scale)
        z_sample = q_z.rsample() # reparameterization

        # Choose p_z as standard normal
        p_z = Normal(torch.zeros((batch_size, self._latent_size), device=x.device),\
            torch.ones((batch_size, self._latent_size), device=x.device))

        # Retrieve p_x parameters given z
        p_x_logits = self.generative_network(z_sample) # (batch, channels, H, W)

        return p_z, q_z, p_x_logits

class InferenceNetwork(nn.Module):
    '''Map x to parameters for the q(z|x) distribution. Convolutions are based on lenet5.'''

    def __init__(self, data_dim, latent_size=1024):
        '''data_dim: (H, W, channels)'''
        super(InferenceNetwork, self).__init__()
        self.data_dim = data_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=data_dim[2], out_channels=32, kernel_size=2, stride=2, padding=0), # 32x48x48
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x25x25
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0), # 64x12x12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0), # 128x6x6
            nn.Flatten()
        )
        self.loc_fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, latent_size),
        )
        self.scale_fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, latent_size),
            nn.Softplus(),
        )

    def forward(self, x):
        ''' x: (batch, channels, H, W)
        Returns: (batch, latent)'''
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # split non-batch dim in half
        loc = self.loc_fc(out)
        scale = self.scale_fc(out)
        return loc, scale

class GenerativeNetwork(nn.Module):
    '''Map latent variables to parameters for the p(x|z) distribution.'''

    def __init__(self, data_dim, latent_size=1024, convs_out_dim=(6, 6, 128)):
        super(GenerativeNetwork, self).__init__()
        self.data_dim = data_dim
        self._convs_out_dim = convs_out_dim

        self.generative1 = nn.Linear(latent_size, np.prod(convs_out_dim)) # Reshape after
        self.generative2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(64),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(32),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(32),

            nn.ConvTranspose2d(in_channels=32, out_channels=data_dim[2], kernel_size=2, stride=2, padding=0),

            nn.Sigmoid() # Sigmoid here works better than Tanh
        )

    def forward(self, z_sample):
        '''z_sample: (batch, latent)
        Returns: (batch, channels, H, W)
        '''
        out = self.generative1(z_sample)
        h, w, channels = self._convs_out_dim
        p_x_logits = self.generative2(out.view(-1, channels, h, w))
        return p_x_logits

if __name__ == '__main__':
    device = torch.device('cuda')
    vae = VAE((96, 96, 1)).to(device)
    inference = InferenceNetwork((96, 96, 1)).to(device)
    generative = GenerativeNetwork((96, 96, 1)).to(device)

    t = torch.ones((102, 1, 96, 96), device=device)
    # loc, _ = inference(t)
    # p_x_logits = generative(loc)
    p_z, q_z, p_x_logits = vae(t)
    print(p_x_logits.shape)
