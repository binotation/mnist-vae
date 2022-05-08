import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers import create_dir, get_data
from sklearn.decomposition import PCA

def check(net, X_ts, img_path):
    sample_size = 10
    r = torch.randint(0, len(X_ts) - sample_size, (1,)).item()
    imgs = X_ts[r : r + sample_size].reshape(sample_size, 28, 28)

    fig, axs = plt.subplots(2, 5, constrained_layout=True)
    for j, img in enumerate(imgs):
        axs[j // 5, j % 5].imshow(img.cpu().detach().numpy())
    plt.savefig(img_path + '/original.png')

    fig, axs = plt.subplots(2, 5, constrained_layout=True)
    for j, img in enumerate(imgs):
        reconstructed = net(img.view(1, 784))
        axs[j // 5, j % 5].imshow(reconstructed.view(28, 28).cpu().detach().numpy())
    plt.savefig(img_path + '/reconstructed.png')

def latent_space(net, X_ts, y_ts, img_path):
    # Register hook to collect the output of enc_out layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    net.enc_out.register_forward_hook(get_activation('enc_out'))
    _ = net(X_ts)
    latent_vecs = activation['enc_out'] # (10000, 72)

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

    img_path = __file__ + '/../../img'
    create_dir(img_path)

    trained_path = __file__ + '/../../trained/'
    trained = os.listdir(trained_path)

    net = torch.load(trained_path + trained[0]).to(device)
    X_tr, X_ts, y_ts = get_data(device)

    check(net, X_ts, img_path)
    latent_space(net, X_ts, y_ts, img_path)

if __name__ == '__main__':
    main()
