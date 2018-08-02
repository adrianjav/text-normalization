import math
import torch.nn as nn
from utils import check_size, reset_parameters

from torch.nn import LSTM  # Encoder 1: Long Short-Term Memory
from cfe import CausalFeatureExtractor  # Encoder 4: Causal Feature Extractor

# Encoder 2: Fully Convolutional Neural Network with width 1


class Conv2dChainer(nn.Module):
    def __init__(self):
        super(Conv2dChainer, self).__init__()

    def forward(self, x):
        return x.squeeze(dim=2).unsqueeze(1)


class FullyConvolutionalNN(nn.Module):
    def __init__(self, input_size, feats_size, num_layers, dropout):
        super(FullyConvolutionalNN, self).__init__()
        self.input_size = input_size
        self.dropout = dropout

        self.conv_layers = [nn.Conv2d(1, feats_size, kernel_size=(input_size, 1))]
        self.conv_layers.extend([nn.Conv2d(1, feats_size, kernel_size=(feats_size, 1)) for _ in range(num_layers)])

        layers = [nn.Conv2d(1, feats_size, kernel_size=(input_size, 1))]
        for i in range(num_layers):
            layers += [
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                Conv2dChainer(),
                nn.Conv2d(1, feats_size, kernel_size=(feats_size, 1))
            ]

        self.net = nn.Sequential(*layers)
        reset_parameters(self, 'conv2d')

    def forward(self, input):
        check_size(input, (-1, self.input_size, -1))  # input b x v x l

        return self.net(input.unsqueeze(1)).squeeze(2)

# Encoder 3: Feature Extractor


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size

        num_layers = max(1, int(math.ceil(math.log2(receptive_field/(kernel_size-1) + 1) - 1)))
        layers = [nn.Conv2d(1, feats_size, kernel_size=(input_size, kernel_size),
                            padding=(0,(kernel_size-1)//2))]
        for i in range(1, num_layers):
            layers += [
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(feats_size, feats_size, kernel_size=(1, kernel_size),
                                 padding=(0, (kernel_size - 1) * 2 ** (i - 1)), dilation=(1, 2 ** i))
            ]
        self.net = nn.Sequential(*layers)

        reset_parameters(self, 'conv2d')

    def forward(self, input):
        check_size(input, (-1, self.input_size, -1))  # input b x v x l

        return self.net(input.unsqueeze(dim=1)).squeeze(dim=2)
