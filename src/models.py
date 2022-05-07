import torch
import torch.nn as nn

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
