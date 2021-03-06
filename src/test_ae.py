import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reconstruct(model, X_ts, img_path):
    '''Create an image of original data vs reconstructed'''
    sample_size = 10
    r = torch.randint(0, len(X_ts) - sample_size, (1,)).item()
    sample = X_ts[r : r + sample_size]

    reconstructed = model(sample).permute((0, 2, 3, 1))

    fig, axs = plt.subplots(2, 10, constrained_layout=True, figsize=(15, 5))
    fig.suptitle('Original (top row) vs Reconstructed')
    for j, img in enumerate(sample.permute((0, 2, 3, 1))):
        axs[0, j].imshow(img.cpu().detach().numpy())

    for j, img in enumerate(reconstructed):
        axs[1, j].imshow(img.cpu().detach().numpy())

    fig.savefig(img_path + '/reconstructed.png')

def latent_space(model, X_ts, y_ts, img_path):
    '''Create a plot of the latent vectors embedded into 2D'''
    latent_vecs = model.encoder(X_ts)

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
    plt.title('AE latent vectors embedded into 2D (latent space)')
    fig.savefig(img_path + '/latent_space.png')
    return pca

def create_new(device, model, img_path, h, channels):
    z = torch.rand((5, 64), device=device) * 10 - 5
    new = model.decoder(z)
    fig, axs = plt.subplots(1, 5, constrained_layout=True, figsize=(8, 3))
    for i in range(5):
        axs[i].imshow(new[i].view(h, h, channels).cpu().detach().numpy())
    fig.suptitle('Uniform random samples of latent space on (-5, 5)')
    fig.savefig(img_path + '/new.png')

def choose_new(device, pca: PCA, model, h, channels, img_path, point):
    '''pca comes from latent_space'''
    z_reduced = np.array(point)
    z = torch.from_numpy(pca.inverse_transform(z_reduced)).unsqueeze(0).float().to(device)

    x = model.decoder(z)
    fig = plt.figure()
    plt.imshow(x.view(h, h, channels).cpu().detach().numpy())
    fig.savefig(img_path + '/choose.png')
