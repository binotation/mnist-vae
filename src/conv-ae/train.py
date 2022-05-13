# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import get_mnist, get_cifar10
from model import ConvAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(device, X_tr, epochs=20):
    shape = X_tr.shape
    loader = DataLoader(X_tr, batch_size=125, shuffle=True)
    conv_ae = ConvAE((shape[2], shape[3], shape[1])).to(device)
    optimizer = optim.Adam(conv_ae.parameters(), 8e-4)
    criterion = nn.MSELoss()

    loop = tqdm(range(epochs))
    for epoch in loop:
        for x in loader:
            optimizer.zero_grad()
            reconstructed = conv_ae(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
        loop.set_postfix(loss=f'{loss:.5f}')

    return conv_ae

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr, y_tr, X_ts, y_ts = get_cifar10(device, root='g:/Personal/blob')

    conv_ae = train(device, X_tr)

    # Save model
    torch.save(conv_ae, f'{__file__}/../conv_ae.pkl')

if __name__ == '__main__':
    main()
