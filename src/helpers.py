import os
import torchvision.datasets as datasets

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_data(device):
    set_tr = datasets.MNIST(root='~/.torchvision', train=True)
    set_ts = datasets.MNIST(root='~/.torchvision', train=False)

    # (60000, 784)
    X_tr = set_tr.data.float().view(len(set_tr), -1).to(device)

    # (10000, 784)
    X_ts = set_ts.data.float().view(len(set_ts), -1).to(device)

    # (10000,)
    y_ts = set_ts.targets

    return X_tr, X_ts, y_ts
