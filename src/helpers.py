import torch
import os
import torchvision.datasets as datasets

def create_dir(path):
    '''Create a directory at path if it does not exist.'''
    if not os.path.exists(path):
        os.makedirs(path)

def get_conv_out_dim(in_dim, out_channels, kernel_size, stride, padding):
    '''Calculate the output dimension for 1 conv2d.
    in_dim: (H, W, channels)
    Returns: (H, W, channels)
    '''
    h = (in_dim[0] - kernel_size + 2*padding) // stride + 1
    w = (in_dim[1] - kernel_size + 2*padding) // stride + 1
    d = out_channels
    return (h, w, d)

def get_convs_out_dim(in_dim, out_channels, kernel_size, stride, padding):
    '''Calculate the output dimension for a sequence of conv2d.
    in_dim: (H, W, channels)
    Other arguments should be (int,int,...) of the same length.
    Returns: (H, W, channels)
    '''
    for i in range(len(out_channels)):
        in_dim = get_conv_out_dim(in_dim, out_channels[i], kernel_size[i], stride[i], padding[i])
    return in_dim

def get_mnist(device, root='~/.torchvision', download=False):
    '''Get mnist data.
        - 10 classes
        - 28 x 28 dim
        - X_tr: 60000 samples
        - X_ts: 10000 samples
        - int (0-255) normalized to float32 (0-1)
        - Data shaped to (batch, channels, H, W)
    '''
    set_tr = datasets.MNIST(root=root, train=True, download=download)
    set_ts = datasets.MNIST(root=root, train=False, download=download)

    # (60000, 28, 28)
    X_tr = (set_tr.data / 255).unsqueeze(1).to(device)

    # (60000,)
    y_tr = set_tr.targets

    # (10000, 28, 28)
    X_ts = (set_ts.data / 255).unsqueeze(1).to(device)

    # (10000,)
    y_ts = set_ts.targets

    return X_tr, y_tr, X_ts, y_ts

def get_cifar10(device, root='~/.torchvision', download=False):
    '''Get cifar10 data.
        - 10 classes
        - 32 x 32 x 3 dim
        - X_tr: 50000 samples
        - X_ts: 10000 samples
        - int (0-255) normalized to float32 (0-1)
        - Data shaped to (batch, channels, H, W)
    '''
    set_tr = datasets.CIFAR10(root=root, train=True, download=download)
    set_ts = datasets.CIFAR10(root=root, train=False, download=download)

    # (50000, 3, 32, 32)
    X_tr = torch.from_numpy(set_tr.data / 255).float().permute(0, 3, 1, 2).to(device)

    # list (60000,)
    y_tr = set_tr.targets

    # (10000, 3, 32, 32)
    X_ts = torch.from_numpy(set_ts.data / 255).float().permute(0, 3, 1, 2).to(device)

    # list (10000,)
    y_ts = set_ts.targets

    return X_tr, y_tr, X_ts, y_ts
