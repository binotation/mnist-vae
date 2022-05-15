# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import create_dir, get_mnist, get_cifar10
from test_ae import reconstruct, latent_space, create_new, choose_new
from model import LinearAE
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = __file__ + '/../img'
    create_dir(img_path)

    linear_ae = torch.load(f'{__file__}/../linear_ae.pkl').to(device)
    X_tr, y_tr, X_ts, y_ts = get_cifar10(device, root='g:/Personal/blob')

    h = X_ts.shape[2]
    channels = X_ts.shape[1]
    reconstruct(linear_ae, X_ts, img_path)
    pca = latent_space(linear_ae, X_ts, y_ts, img_path)
    create_new(device, linear_ae, img_path, h, channels)
    choose_new(device, pca, linear_ae, h, channels, img_path, (-5, -8))

if __name__ == '__main__':
    main()
