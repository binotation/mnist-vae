# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import create_dir, get_cifar10
from test_ae import reconstruct, latent_space
from model import ConvAE
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = __file__ + '/../img'
    create_dir(img_path)

    conv_ae = torch.load(f'{__file__}/../conv_ae.pkl').to(device)
    X_tr, y_tr, X_ts, y_ts = get_cifar10(device, root='g:/Personal/blob')

    reconstruct(conv_ae, X_ts, img_path)
    latent_space(conv_ae, X_ts, y_ts, img_path)

if __name__ == '__main__':
    main()
