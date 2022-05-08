import torch
import torch.optim as optim
import torch.nn as nn
from helpers import create_dir, get_data
from models import LinearAE, ConvAE
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    create_dir(trained_path)
    torch.save(net, f'{trained_path}/{name}_{loss:.0f}.pkl')
    return net

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr, X_ts, y_ts = get_data(device)

    # train(LinearAE(X_tr.shape[1], 256, 72).to(device),\
    #     'LinearAE',\
    #     X_tr,\
    #     20,\
    #     nn.MSELoss(),\
    #     lambda p: optim.Adam(p, 8e-4))

    train(ConvAE(X_tr.shape[1], 256, 48).to(device),\
        'ConvAE',\
        X_tr,\
        20,\
        nn.MSELoss(),\
        lambda p: optim.Adam(p, 8e-4))

if __name__ == '__main__':
    main()
