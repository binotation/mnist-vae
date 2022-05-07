import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAE(nn.Module):
    def __init__(self, in_size, hid_size):
        super(LinearAE, self).__init__()
        self.enc_in = nn.Linear(in_size, hid_size)
        self.enc_out = nn.Linear(hid_size, hid_size)
        self.dec_in = nn.Linear(hid_size, hid_size)
        self.dec_out = nn.Linear(hid_size, in_size)

    def forward(self, input):
        x = self.enc_in(input)
        x = self.enc_out(x)
        x = self.dec_in(x)
        x = self.dec_out(x)
        x = torch.relu(x)
        return x

class ConvAE(nn.Module):
    def __init__(self, in_size, hid_size):
        super(ConvAE, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3, 1)
        self.enc_in = nn.Linear(5408, hid_size)
        self.enc_out = nn.Linear(hid_size, hid_size)
        self.dec_in = nn.Linear(hid_size, hid_size)
        self.dec_out = nn.Linear(hid_size, in_size)

    def forward(self, input):
        x = F.max_pool2d(self.conv(input.view(-1, 1, 28, 28)), 2)
        x = torch.flatten(x, 1)
        x = self.enc_in(x)
        x = self.enc_out(x)
        x = self.dec_in(x)
        x = F.relu(self.dec_out(x))
        return x
