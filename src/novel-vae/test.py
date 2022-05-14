# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import create_dir, get_pokemon
from model import VAE
import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

def sample(device, vae, img_path):
    z = torch.randn((10, 512), device=device)
    out = vae.generative_network(z).permute((0, 2, 3, 1))
    fig, axs = plt.subplots(1, 10, constrained_layout=True, figsize=(15, 3))
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
    p_x_logits = vae.generative_network(z_sample)
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 3))
    for j, img in enumerate(p_x_logits.permute((0, 2, 3, 1))):
        axs[j].imshow(img.cpu().detach().numpy())
    fig.savefig(f'{img_path}/reconstructed.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = __file__ + '/../img'
    create_dir(img_path)

    imgs = get_pokemon(device)

    vae = torch.load(f'{__file__}/../vae.pkl').to(device)
    sample(device, vae, img_path)
    reconstruct(vae, imgs, img_path)

if __name__ == '__main__':
    main()
