import torch.nn as nn
import encoders as encs
from utils import check_size, reset_parameters


class Attention(nn.Module):
    def __init__(self, hidden_size, feats_size, ctx_size, num_mid_layer):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.feats_size = feats_size

        self.net = nn.Sequential(
            nn.Linear(hidden_size, num_mid_layer), nn.LeakyReLU(),
            nn.Linear(num_mid_layer, num_mid_layer), nn.LeakyReLU(),
            nn.Linear(num_mid_layer, feats_size)
        )
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveMaxPool1d(output_size=ctx_size)

        reset_parameters(self, 'leaky_relu')

    def forward(self, hidden, feats):
        check_size(hidden, (-1, self.hidden_size))  # hidden b x h
        check_size(feats, (hidden.size()[0], self.feats_size, -1))  # feats b x f x l

        attn = self.net(hidden.unsqueeze(1))  # b x 1 x f
        attn = attn.squeeze(1).unsqueeze(2)  # b x f x 1
        attn = attn * feats  # b x f x l
        attn = attn.sum(dim=1)  # b x l
        attn = self.softmax(attn)  # b x l

        return self.pool(feats * attn.unsqueeze(1)), attn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size

        self.input_weights = nn.Linear(input_size, 4*hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        self.ctx_weights = nn.Linear(ctx_size, 4 * hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        reset_parameters(self)

    def forward(self, input, hidden, ctx):
        check_size(input, (-1, self.input_size))  # input b x v
        check_size(hidden, (input.size()[0], self.hidden_size))  # hidden b x h
        check_size(ctx, (input.size()[0], self.ctx_size))  # ctx b x (f * c)

        hx, cx = hidden  # n_b x hidden_dim
        gates = self.input_weights(input) + self.hidden_weights(hx) + self.ctx_weights(ctx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.sigmoid(ingate)
        forgetgate = self.sigmoid(forgetgate)
        cellgate = self.tanh(cellgate)
        outgate = self.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.tanh(cy)

        return hy, cy


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, feats_size, ctx_size, hidden_size, num_layers, dropout):
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size
        self.feats_size = feats_size
        self.ctx_size = ctx_size
        self.hidden_size = hidden_size

        self.layers = [LSTMCell(input_size, hidden_size, feats_size * ctx_size)]
        self.layers += [LSTMCell(hidden_size, hidden_size, feats_size * ctx_size) for _ in range(1, num_layers)]

        for i, layer in enumerate(self.layers[:-1]):
            self.add_module('lstm{}'.format(i), layer)
            self.add_module('dropout{}'.format(i), nn.Dropout(dropout))

        self.add_module('lstm{}'.format(len(self.layers)-1), self.layers[-1])

    def forward(self, input, ctx, hidden, cell):
        check_size(input, (-1, self.input_size))  # input b x v
        check_size(ctx, (input.size()[0], self.feats_size, self.ctx_size))  # ctx b x f x c
        check_size(hidden, (input.size()[0], self.hidden_size))  # hidden tuple of b x h
        check_size(cell, (input.size()[0], self.hidden_size))  # cell tuple of b x h

        ctx = ctx.reshape((ctx.size()[0], -1))
        hidden_states = []
        for i, layer in enumerate(self.layers):
            hidden_states.append(layer(input, (hidden[i], cell[i]), ctx))
            input = hidden_states[-1][0]

        return zip(*hidden_states)


class Output(nn.Module):
    def __init__(self, input_size, hidden_size, n_linear_hidden):
        super(Output, self).__init__()
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(hidden_size, n_linear_hidden), nn.LeakyReLU(),
            nn.Linear(n_linear_hidden, n_linear_hidden), nn.LeakyReLU(),
            nn.Linear(n_linear_hidden, input_size),
            nn.LogSoftmax(dim=1)
        )

        reset_parameters(self, 'leaky_relu')

    def forward(self, input):
        check_size(input, (-1, self.hidden_size))  # input b x h

        return self.net(input)


class InitialWeights(nn.Module):
    def __init__(self, hidden_size, num_linear_hidden, lstm_layers):
        super(InitialWeights, self).__init__()
        self.lstm_layers = lstm_layers

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.net = nn.Sequential(
            nn.Linear(1, num_linear_hidden), nn.LeakyReLU(),
            nn.Linear(num_linear_hidden, num_linear_hidden), nn.LeakyReLU(),
            nn.Linear(num_linear_hidden, hidden_size * 2 * lstm_layers)
        )

        reset_parameters(self, 'leaky_relu')

    def forward(self, feats):
        check_size(feats, (-1, -1, -1))  # feats b x f x l

        feats = self.pool(feats).squeeze(1)
        out = self.net(feats).chunk(2, dim=1)
        out = [x.chunk(self.lstm_layers, dim=1) for x in out]
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout, which):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.which = which

        self.net = {
            'lstm': encs.LSTM(input_size, feats_size // 2, 3, batch_first=True, dropout=dropout, bidirectional=True),
            'dfc': encs.FullyConvolutionalNN(input_size, feats_size, 4, dropout),
            'fe': encs.FeatureExtractor(input_size, feats_size, kernel_size, receptive_field, dropout),
            'cfe': encs.CausalFeatureExtractor(input_size, feats_size, kernel_size, receptive_field, dropout)
        }[which]

    def forward(self, input):
        check_size(input, (-1, self.input_size, -1))  # input b x v x l

        if self.which == 'lstm':
            out, _ = self.net(input.transpose(1,2))
            return out.transpose(1,2)
        return self.net(input)


class Decoder(nn.Module):
    def __init__(self, input_size, feats_size, ctx_size, hidden_size, num_mid_layer, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.feats_size = feats_size
        self.ctx_size = ctx_size
        self.hidden_size = hidden_size

        self.attention = Attention(hidden_size, feats_size, ctx_size, num_mid_layer)
        self.lstm = AttentionLSTM(input_size, feats_size, ctx_size, hidden_size, 4, dropout)
        self.output = Output(input_size, hidden_size, num_mid_layer)

        self.tolog = nn.LogSoftmax(dim=1)

    def forward(self, input, feats, hidden, cell):
        check_size(input, (-1, self.input_size))  # input b x v
        check_size(feats, (input.size()[0], self.feats_size, -1))  # feats b x f x l
        check_size((hidden, cell), (input.size()[0], self.hidden_size))  # hidden b x h

        ctx, attn = self.attention(hidden[-1], feats)

        hidden, cell = self.lstm(input, ctx, hidden, cell)
        out = self.output(hidden[-1])

        return out, attn, hidden, cell
