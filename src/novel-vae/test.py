# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import create_dir, get_pokemon_grayscale
from model import VAE
import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

def sample(device, vae, img_path):
    z = torch.randn((5, 2048), device=device)
    out = vae.generative_network(z).permute((0, 2, 3, 1))
    fig, axs = plt.subplots(1, 5, constrained_layout=True, figsize=(15, 3))
    for j, img in enumerate(out):
        axs[j].imshow(img.cpu().detach().numpy())
    plt.savefig(f'{img_path}/sample.png')

def reconstruct(vae, imgs, img_path):
    sample_size = 2
    start = torch.randint(0, 750 - sample_size, (1,)).item()
    sample = imgs[start: start + sample_size]
    q_z_loc, q_z_scale = vae.inference_network(sample)
    q_z = Normal(q_z_loc, q_z_scale)
    z_sample = q_z.sample()
    out = vae.generative_network(z_sample)
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(6, 6))
    for j, img in enumerate(sample.squeeze(1)):
        axs[0][j].imshow(img.cpu().detach().numpy())
    for j, img in enumerate(out.permute((0, 2, 3, 1))):
        axs[1][j].imshow(img.cpu().detach().numpy())
    mid_q = Normal(q_z_loc.sum(0, keepdim=True), q_z_scale.sum(0, keepdim=True))
    z_sample = mid_q.sample()
    out = vae.generative_network(z_sample).permute(0, 2, 3, 1)
    axs[1][2].imshow(out[0].cpu().detach().numpy())
    axs[0][2].set_visible(False)
    fig.savefig(f'{img_path}/reconstructed.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = __file__ + '/../img'
    create_dir(img_path)

    imgs = get_pokemon_grayscale(device)

    vae = torch.load(f'{__file__}/../vae.pkl').to(device)
    sample(device, vae, img_path)
    reconstruct(vae, imgs, img_path)

if __name__ == '__main__':
    main()
