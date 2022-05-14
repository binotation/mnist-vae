# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import create_dir, get_mnist, get_cifar10
from model import VAE
import torch
import matplotlib.pyplot as plt

def sample(device, vae, img_path):
    z = torch.randn((10, 128), device=device)
    out = vae.generative_network(z).squeeze(1)
    fig, axs = plt.subplots(1, 10, constrained_layout=True, figsize=(15, 3))
    for j, img in enumerate(out):
        axs[j].imshow(img.cpu().detach().numpy())
    plt.savefig(f'{img_path}/sample.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = __file__ + '/../img'
    create_dir(img_path)

    vae = torch.load(f'{__file__}/../vae.pkl').to(device)
    sample(device, vae, img_path)

if __name__ == '__main__':
    main()
