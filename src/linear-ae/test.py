# Allow importing from parent directory
import sys
sys.path.append(__file__ + '/../..')

from helpers import get_mnist, create_dir
from test_ae import reconstruct, latent_space
from model import LinearAE
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = __file__ + '/../img'
    create_dir(img_path)

    linear_ae = torch.load(f'{__file__}/../linear_ae.pkl').to(device)
    X_tr, y_tr, X_ts, y_ts = get_mnist(device)

    reconstruct(linear_ae, X_ts, img_path)
    latent_space(linear_ae, X_ts, y_ts, img_path)

if __name__ == '__main__':
    main()
