from modules import Encoder, Decoder, InitialWeights
import globals as _g
import torch


class TextNormalizer(torch.nn.Module):
    def __init__(self, vocab, feats_size, kernel_size, rec_field, attn_size, hidden_size, mid_layer, dropout, which):
        super(TextNormalizer, self).__init__()
        self.vocab = vocab
        self.encoder = Encoder(len(vocab), feats_size, kernel_size, rec_field, dropout, which)
        self.decoder = Decoder(len(vocab), feats_size, attn_size, hidden_size, mid_layer, dropout)
        self.init_hidden = InitialWeights(hidden_size, mid_layer, 4)

    def train(self, batch, seq_len=None):
        return ForgetfulIterator(self, batch, seq_len)

    def eval(self, batch, seq_len=None):
        return ForgetfulIterator(self, batch, seq_len)


class ForgetfulIterator(object):
    def __init__(self, model, data, seq_len=None):
        self.model = model
        self.data = data
        self.seq_len = seq_len
        self.pos = 0

        self.max_seq_len = 2000

    def __len__(self):
        return self.seq_len

    def __enter__(self):
        self.feats = self.model.encoder(self.data)
        self.inp = self.data[..., 0]
        self.hidden, self.cell = self.model.init_hidden(self.feats)
        self.pos = 1
        self.finished = self.data.new_zeros(self.data.size()[0], dtype=torch.uint8)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.hidden, self.cell, self.feats, self.inp
        return False

    def __iter__(self):
        return self

    def __next__(self):
        if self.seq_len is not None:
            if self.pos == self.seq_len:
                raise StopIteration
        elif (self.finished != 0).all() or self.pos == self.max_seq_len:  # ensure stopping
            raise StopIteration

        self.inp = torch.exp(self.inp)
        self.inp, self.attn, self.hidden, self.cell = self.model.decoder(self.inp, self.feats, self.hidden, self.cell)
        if self.seq_len is None:
            self.finished |= torch.max(self.inp, dim=1)[1] == self.model.vocab.stoi[_g.end_symbol]

        self.pos += 1
        return self.inp


class TextNormalizerWithBeam(torch.nn.Module):
    def __init__(self, vocab, feats_size, kernel_size, rec_field, attn_size, hidden_size, mid_layer, dropout, which,
                 beam_size):
        super(TextNormalizerWithBeam, self).__init__()
        self.vocab = vocab
        self.encoder = Encoder(len(vocab), feats_size, kernel_size, rec_field, dropout, which)
        self.decoder = Decoder(len(vocab), feats_size, attn_size, hidden_size, mid_layer, dropout)
        self.init_hidden = InitialWeights(hidden_size, mid_layer, 4)
        self.beam_size = beam_size

    def train(self, batch, seq_len=None):
        return ForgetfulIterator(self, batch, seq_len)

    def eval(self, batch, seq_len=None):
        return BeamIterator(self, batch, seq_len, self.beam_size)


class BeamIterator(object):
    def __init__(self, model, data, beam_size):
        raise NotImplementedError

