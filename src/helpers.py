import os
import torchvision.datasets as datasets

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_mnist(device, root='~/.torchvision', download=False):
    '''Get mnist data.
        - 10 classes
        - 28 x 28 dim
        - X_tr: 60000 samples
        - X_ts: 10000 samples
    '''
    set_tr = datasets.MNIST(root=root, train=True, download=download)
    set_ts = datasets.MNIST(root=root, train=False, download=download)

    # (60000, 28, 28)
    X_tr = (set_tr.data / 255).to(device)

    # (60000,)
    y_tr = set_tr.targets

    # (10000, 28, 28)
    X_ts = (set_ts.data / 255).to(device)

    # (10000,)
    y_ts = set_ts.targets

    return X_tr, y_tr, X_ts, y_ts
