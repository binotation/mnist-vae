# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import get_mnist, get_cifar10
from model import LinearAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(device, X_tr, epochs=20):
    shape = X_tr.shape
    loader = DataLoader(X_tr, batch_size=125, shuffle=True)
    linear_ae = LinearAE((shape[2], shape[3], shape[1])).to(device)
    optimizer = optim.Adam(linear_ae.parameters(), 1e-4)
    criterion = nn.MSELoss()

    loop = tqdm(range(epochs))
    for epoch in loop:
        for x in loader:
            optimizer.zero_grad()
            reconstructed = linear_ae(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
        loop.set_postfix(loss=f'{loss:.5f}')

    return linear_ae

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr, y_tr, X_ts, y_ts = get_cifar10(device, root='g:/Personal/blob')

    linear_ae = train(device, X_tr)

    # Save model
    torch.save(linear_ae, f'{__file__}/../linear_ae.pkl')

if __name__ == '__main__':
    main()
