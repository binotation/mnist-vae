# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from helpers import get_convs_out_dim

class VAE(nn.Module):
    '''Variational autoencoder. p_x is chosen as Bernoulli.'''
    def __init__(self, data_dim):
        super(VAE, self).__init__()
        self._latent_size = 512
        self._convs_out_dim = 512
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

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=data_dim[2], out_channels=64, kernel_size=4, stride=2, padding=3), # 50x50
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential( # 26x26
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential( # 13x13
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv4 = nn.Sequential( # 7x7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv5 = nn.Sequential( # 4x4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(4), # 1x1
            nn.Flatten()
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
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
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

        self.generative1 = nn.Linear(latent_size, 512) # Reshape after
        self.generative2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=3, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=data_dim[2], kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(data_dim[2]),
            nn.Sigmoid() # Sigmoid here works better than Tanh
        )

    def forward(self, z_sample):
        '''z_sample: (batch, latent)
        Returns: (batch, channels, H, W)
        '''
        out = self.generative1(z_sample)
        p_x_logits = self.generative2(out.view(-1, 512, 1, 1))
        return p_x_logits

if __name__ == '__main__':
    inference = InferenceNetwork((96, 96, 3), 128, 512)
    vae = VAE((96, 96, 3))
    t = torch.ones((102, 3, 96, 96))
    # loc, _ = inference(t)
    p_z, q_z, p_x_logits = vae(t)
    print(p_x_logits.shape)
