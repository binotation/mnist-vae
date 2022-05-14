# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from model import VAE
from helpers import get_pokemon
from torch.utils.data import DataLoader
from tqdm import tqdm

def negative_elbo(p_x_logits, x, q_z, p_z):
    '''Negative elbo loss function. 
    elbo = log likelihood of x under p_x - Kullback Leibler divergence
    Since p_x is Bernoulli, the first term is equivalent to the negative binary cross entropy loss.
    '''
    # Log prob of x under p_x equivalent to binary cross entropy loss
    # We don't use binary cross entropy with logits because the last layer of the generative network
    # has sigmoid activation
    bce = F.binary_cross_entropy(p_x_logits, x, reduction='sum')

    kl = distributions.kl.kl_divergence(q_z, p_z).sum()

    # -elbo = -(-bce - kl) = bce + kl
    return bce + kl

def train(device, X_tr, epochs=100):
    batches, channels, h, w = X_tr.shape
    vae = VAE((h, w, channels)).to(device)

    loader = DataLoader(X_tr, batch_size=50, shuffle=True)
    optimizer = optim.Adam(vae.parameters(), lr=2e-3)

    loop = tqdm(range(epochs))
    for epoch in loop:
        for x in loader:
            optimizer.zero_grad()
            p_z, q_z, p_x_logits = vae(x)
            loss = negative_elbo(p_x_logits, x, q_z, p_z)
            loss.backward()
            optimizer.step()
        loop.set_postfix(loss=f'{loss:7.5f}')
    return vae

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs = get_pokemon(device)

    vae = train(device, imgs)

    # Save model
    torch.save(vae, f'{__file__}/../vae.pkl')

if __name__ == '__main__':
    main()
