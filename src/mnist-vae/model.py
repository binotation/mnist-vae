import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from helpers import get_convs_out_dim

class VAE(nn.Module):
    '''Variational autoencoder. p_x is chosen as Bernoulli.'''
    def __init__(self, data_dim):
        super(VAE, self).__init__()
        self._latent_size = 128
        self._convs_out_dim = get_convs_out_dim(data_dim, (6, 6, 16, 16), (5, 2, 5, 2), (1, 2, 1, 2), (0,)*4)
        self.inference_network = InferenceNetwork(data_dim, self._latent_size, np.prod(self._convs_out_dim))
        self.generative_network = GenerativeNetwork(data_dim, self._latent_size, self._convs_out_dim)

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

    def __init__(self, data_dim, latent_size, convs_out_size):
        '''data_dim: (H, W, channels)'''
        super(InferenceNetwork, self).__init__()
        self.data_dim = data_dim

        self.inference = nn.Sequential(
            nn.Conv2d(in_channels=data_dim[2], out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Flatten(),
        )
        self.loc_fc = nn.Sequential(
            nn.Linear(convs_out_size, latent_size),
        )
        self.scale_fc = nn.Sequential(
            nn.Linear(convs_out_size, latent_size),
            nn.Softplus(),
        )

    def forward(self, x):
        ''' x: (batch, channels, H, W)
        Returns: (batch, latent)'''
        out = self.inference(x)
        # split non-batch dim in half
        loc = self.loc_fc(out)
        scale = self.scale_fc(out)
        return loc, scale

class GenerativeNetwork(nn.Module):
    '''Map latent variables to parameters for the p(x|z) distribution.'''

    def __init__(self, data_dim, latent_size, convs_out_dim):
        super(GenerativeNetwork, self).__init__()
        self.data_dim = data_dim
        self._convs_out_dim = convs_out_dim

        self.generative1 = nn.Linear(latent_size, np.prod(convs_out_dim)) # Reshape after
        self.generative2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=2, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=6, out_channels=data_dim[2], kernel_size=5),
            nn.BatchNorm2d(data_dim[2]),
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
