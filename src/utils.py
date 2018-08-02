import torch
import torch.nn as nn
import globals as _g


def check_size(tensor, size):
    if _g.args.debug:
        if not isinstance(size, (list, tuple)):
            size = [size]

        if isinstance(tensor, (list, tuple)):
            for t in tensor:
                check_size(t, size)
        else:
            assert len(size) == len(tensor.size()), "Different number of dimensions! Got {}, but {} were expected".format(
                tuple(tensor.size()), size
            )

            for i, (x, y) in enumerate(zip(tensor.size(), size)):
                assert y == -1 or x == y, "The dimension {} has wrong size. Got {}, but expected {}".format(i, x, y)


def reset_parameters(module, gain='linear'):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain(gain))


def to_one_hot(var, vocab_size):
    if len(var.size()) == 1:  # to infer cases
        var = var.unsqueeze(1)

    out = var.new_zeros(var.size()[0], vocab_size, var.size()[1], dtype=torch.float)
    out.scatter_(1, var.unsqueeze(1), 1)
    return out


def to_builtin(x):
    if isinstance(x, dict):
        return {n: to_builtin(t) for n, t in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(t) for t in x]
    if isinstance(x, torch.Tensor):
        return x.item()

    raise AssertionError


def repackage_variable(*args):
    """Wraps hidden states in new Variables, to detach them from their history."""
    args = args[0] if len(args) == 1 else args
    if type(args) == torch.Tensor:
        r = args.detach()
        r.requires_grad_(args.requires_grad)
        return r
    else:
        return [repackage_variable(v) for v in args]


def add_gradient(u, v):
    if type(u) == torch.Tensor:
        u.grad = v.grad if u.grad is None else u.grad if v.grad is None else u.grad + v.grad
    else:
        for x, y in zip(u,v):
            add_gradient(x, y)


def pretty_print_time(x):
    secs = x%60
    x = int(x)
    x //= 60
    mins = x%60
    x //= 60
    hours = x%24
    days = x//24

    out = ''
    do_print = days != 0
    out = out + '{:d}d'.format(days) if do_print else out
    do_print = do_print or hours != 0
    out = out + '{:02d}h'.format(hours) if do_print else out
    do_print = do_print or mins != 0
    out = out + '{:02d}m'.format(mins) if do_print else out
    do_print = do_print or secs != 0
    out = out + '{:2.3f}s'.format(secs) if do_print else out

    return out if out != '' else '0.0s'


def pretty_print(epochs, duration, epoch, total_time, epoch_time, train_loss, eval_measures):
    eta_time = pretty_print_time(
        total_time * (epochs / epoch - 1) if duration is None else duration - total_time
    )
    total_time = pretty_print_time(total_time)
    epoch_time = pretty_print_time(epoch_time)

    text = '| epoch {:d} of {} | epoch time ' + epoch_time + ' | total time ' + total_time + \
           ' | ETA ' + eta_time + ' | train loss {:5.6f}({:5.6f}) |'
    text = text.format(epoch, epochs if duration is None else '?', train_loss[0], train_loss[1])

    for i, j in eval_measures.items():
        text += ' eval ' + i + ' {:5.6f}({:5.6f}) |'
        text = text.format(eval_measures[i][0], eval_measures[i][1])

    print('-'*len(text))
    print(text)
    print('-'*len(text))
