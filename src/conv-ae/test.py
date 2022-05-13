# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import get_mnist, create_dir
from model import ConvAE
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reconstruct(conv_ae, X_ts, img_path):
    sample_size = 10
    r = torch.randint(0, len(X_ts) - sample_size, (1,)).item()
    sample = X_ts[r : r + sample_size].unsqueeze(1)

    reconstructed = conv_ae(sample).squeeze(1)

    fig, axs = plt.subplots(4, 5, constrained_layout=True)
    for j, img in enumerate(sample.squeeze(1)):
        axs[j // 5, j % 5].imshow(img.cpu().detach().numpy())

    for j, img in enumerate(reconstructed):
        j = j + 10
        axs[j // 5, j % 5].imshow(img.cpu().detach().numpy())

    plt.savefig(img_path + '/reconstructed.png')

def latent_space(conv_ae, X_ts, y_ts, img_path):
    latent_vecs = conv_ae.encoder(X_ts.unsqueeze(1))

    # Project latent vector into 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latent_vecs.cpu().detach().numpy())

    # Plot latent space
    plots = []
    fig = plt.figure()
    for j in range(10):
        row_indexes = np.where(y_ts == j) # Get indexes of rows of MNIST class j
        class_rows = reduced[row_indexes]
        plots.append(plt.scatter(class_rows[:, 0], class_rows[:, 1], s=[2] * len(class_rows)))
    lgd = fig.legend(plots, [str(i) for i in range(10)], loc='right')
    for handle in lgd.legendHandles:
        handle.set_sizes((10.0,)) # Change size of colored dot in legend
    plt.title('AE latent vectors projected into 2D (latent space)')
    plt.savefig(img_path + '/latent_space.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = __file__ + '/../img'
    create_dir(img_path)

    conv_ae = torch.load(f'{__file__}/../conv_ae.pkl').to(device)
    X_tr, y_tr, X_ts, y_ts = get_mnist(device)

    reconstruct(conv_ae, X_ts, img_path)
    latent_space(conv_ae, X_ts, y_ts, img_path)

if __name__ == '__main__':
    main()
