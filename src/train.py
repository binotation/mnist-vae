import os
import torch
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from models import LinearAE, ConvAE
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_data(device):
    set_tr = datasets.MNIST(root='~/.torchvision', train=True)
    set_ts = datasets.MNIST(root='~/.torchvision', train=False)

    # (60000, 784)
    X_tr = set_tr.data.float().view(len(set_tr), -1).to(device)

    # (10000, 784)
    X_ts = set_ts.data.float().view(len(set_ts), -1).to(device)

    return X_tr, X_ts

def train(net,\
        name,\
        X_tr,\
        epochs=13,\
        criterion=nn.MSELoss(),\
        optimizer=lambda p: optim.SGD(p, 1e-4)):

    loader = DataLoader(X_tr, batch_size=128, shuffle=True)

    optimizer = optimizer(net.parameters())

    loop = tqdm(range(epochs))
    for _ in loop:
        for x in loader:
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
        loop.set_postfix(loss=f'{loss:7.5f}')

    trained_path = __file__ + '/../../trained'
    if not os.path.exists(trained_path):
        os.makedirs(trained_path)
    torch.save(net, f'{trained_path}/{name}_{loss:.0f}.pkl')
    return net

def check(net, X_ts):
    img_path = __file__ + '/../../img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    sample_size = 10
    r = torch.randint(0, len(X_ts) - sample_size, (1,)).item()
    imgs = X_ts[r : r + sample_size].reshape(sample_size, 28, 28)

    _, axs = plt.subplots(2, 5, constrained_layout=True)
    for j, img in enumerate(imgs):
        axs[j // 5, j % 5].imshow(img.cpu().detach().numpy())
    plt.savefig('img/original.png')

    _, axs = plt.subplots(2, 5, constrained_layout=True)
    for j, img in enumerate(imgs):
        reconstructed = net(img.view(1, 784))
        axs[j // 5, j % 5].imshow(reconstructed.view(28, 28).cpu().detach().numpy())
    plt.savefig('img/reconstructed.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr, X_ts = get_data(device)

    # net = train(LinearAE(X_tr.shape[1], 72).to(device),\
    #     'LinearAE',\
    #     X_tr,\
    #     20,\
    #     nn.MSELoss(),\
    #     lambda p: optim.Adam(p, 8e-4))

    net = train(ConvAE(X_tr.shape[1], 48).to(device),\
        'ConvAE',\
        X_tr,\
        20,\
        nn.MSELoss(),\
        lambda p: optim.Adam(p, 8e-4))

    check(net, X_ts)

if __name__ == '__main__':
    main()
