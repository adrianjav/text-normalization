import argparse
import torch

padding_symbol = '\032'
start_symbol = '\02'
end_symbol = '\03'

vocab = None

parser = argparse.ArgumentParser('Character-level seq2seq model for text normalization (MsC thesis)')
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--quiet', action='store_true', help='quiet mode')
parser.add_argument('--toy', action='store_true', help='activates toy hyperparameters')
parser.add_argument('--load', action='store_true', help='loads the model from a file instead of training it')
parser.add_argument('--train', type=str, default=None, help='location of the training dataset')
parser.add_argument('--val', type=str, default=None, help='location of the validation dataset')
parser.add_argument('--test', type=str, default=None, help='location of the testing dataset')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--dropout', type=float, default=0.95)
parser.add_argument('--rf', type=int, default=20, help='receptive field of the CNN encoders')
parser.add_argument('--nf', type=int, default=20, help='number of features of the encoded matrix')
parser.add_argument('--ks', type=int, default=5, help='kernel size of the CNN encoders')
parser.add_argument('--attn', type=int, default=10, help='number of inputs to consider by the decoder')
parser.add_argument('--hidden', type=int, default=30, help='dimensions of the hidden vectors')
parser.add_argument('--midlayer', type=int, default=30, help='number of neurons of the linear mid layers')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--path', type=str, default='.', help='path where to save the model')
parser.add_argument('--filename', type=str, default='model.pt', help='file name where to load/save the model')
parser.add_argument('--teacher', type=float, default=0.8, help='odds to put the ground-truth as next input')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')  # TODO change to iterations
parser.add_argument('--time', type=int, default=None, help='number of seconds of the training (optional)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=None, help='value of the random seed')
parser.add_argument('--k1', type=int, default=30, help='number of forward-pass timesteps between updates')
parser.add_argument('--k2', type=int, default=40, help='number of timesteps to which apply BPTT')
parser.add_argument('--print-every', type=int, default=1)
parser.add_argument('--clip', type=float, default=5, help='value to which the gradients will be clipped')
parser.add_argument('--examples', type=int, default=0, help='number of examples to infer')
parser.add_argument('--notest', action='store_true', help='avoids testing the model')
parser.add_argument('--input', action='store_true', help='allows the user to insert the cases')
parser.add_argument('--decay', type=int, default=400, help='steps before reducing the learning rate')
parser.add_argument('--decay-factor', type=float, default=0.85, help='value such that lr = value * lr')
parser.add_argument('--encoder', type=str, default='cfe', help='encoder to use: cfe, fe, dfc, lstm')
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
args.teacher = args.teacher if args.teacher else 0.
