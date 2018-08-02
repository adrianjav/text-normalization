import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from utils import check_size, reset_parameters


class Chomp(nn.Module):
    def __init__(self, chomp_size, right=True):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size
        self.right = right

    def forward(self, x):
        return x[..., :-self.chomp_size] if self.right else x[..., self.chomp_size:]


class CausalConv2d(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout, right):
        super(CausalConv2d, self).__init__()
        self.input_size = input_size
        self.feats_size = feats_size
        self.receptive_field = receptive_field
        self.dropout = dropout
        self.right = right

        num_layers = max(1, int(math.ceil(math.log2(receptive_field / (kernel_size - 1) + 1) - 1)))
        layers = [
            weight_norm(nn.Conv2d(1, feats_size, kernel_size=(input_size, kernel_size),
                                  padding=(0, (kernel_size - 1)))),
            Chomp(kernel_size - 1, right)
        ]

        for i in range(1, num_layers):
            layers += [
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                weight_norm(nn.Conv2d(feats_size, feats_size, kernel_size=(1, kernel_size),
                                      padding=(0, (kernel_size - 1) * 2**i), dilation=(1, 2**i))),
                Chomp((kernel_size - 1) * 2**i, right)
            ]

        self.net = nn.Sequential(*layers)
        reset_parameters(self, 'conv2d')

    def forward(self, input):
        check_size(input, (-1, self.input_size, -1))  # input b x v x l

        return self.net(input.unsqueeze(dim=1)).squeeze(dim=2)


class CausalFeatureExtractor(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout):
        super(CausalFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        assert feats_size % 2 == 0, "the number of features has to be an even number"

        self.right = CausalConv2d(input_size, feats_size // 2, kernel_size, receptive_field, dropout, right=True)
        self.left = CausalConv2d(input_size, feats_size // 2, kernel_size, receptive_field, dropout, right=False)

    def forward(self, input):
        check_size(input, (-1, self.input_size, -1))  # input b x v x l

        return torch.cat((self.left(input), self.right(input)), dim=1)
