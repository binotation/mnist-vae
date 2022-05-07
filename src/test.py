import os
import torch
from train import check, get_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_path = __file__ + '/../../trained/'
    trained = os.listdir(trained_path)
    net = torch.load(trained_path + trained[0]).to(device)
    _, X_ts = get_data(device)

    check(net, X_ts)

if __name__ == '__main__':
    main()
