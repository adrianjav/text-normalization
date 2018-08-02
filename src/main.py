import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as torchdata
import numpy as np
import globals as _g
import training as _t
import utils as _u
import inference as _i
from measures import CERLoss, Accuracy
from TextNormalizer import TextNormalizer


def do_train(model, tdata, vdata, measures):
    start_time = time.time()
    input_size = len(_g.vocab)

    # TODO paramatersanitychecker

    torch.save(model, _g.args.path + '/' + _g.args.filename)
    if not _g.args.quiet:
        print('Training...')

    optimizer = optim.Adam(model.parameters(), lr=_g.args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, _g.args.decay, _g.args.decay_factor)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=30)
    criterion = nn.NLLLoss(ignore_index=_g.vocab.stoi[_g.padding_symbol], reduce=False)

    best_eval_loss = None
    t_prints, e_prints = None, None
    epoch_times = None
    saving = False

    try:
        for epoch, (tbatch, vbatch) in enumerate(zip(tdata, vdata)):
            if _g.args.time:
                if time.time() - start_time > _g.args.time:
                    break
            elif epoch == _g.args.epochs:
                break

            epoch_start_time = time.time()
            t_losses = _t.tbtt(model, criterion, optimizer, _u.to_one_hot(tbatch.before, input_size),
                                            tbatch.after)
            t_losses = t_losses.unsqueeze(dim=1)

            e_losses = _t.evaluate(model, criterion, _u.to_one_hot(vbatch.before, input_size), vbatch.after, measures)
            e_losses = e_losses.unsqueeze(dim=1)

            t_prints = t_losses if t_prints is None else torch.cat((t_prints, t_losses), dim=1)
            e_prints = e_losses if e_prints is None else torch.cat((e_prints, e_losses), dim=1)

            epoch_end_time = time.time()
            epoch_time = torch.tensor(epoch_end_time - epoch_start_time)
            epoch_times = epoch_time if epoch_times is None else torch.stack((epoch_times, epoch_time), dim=0)

            if (epoch + 1) % _g.args.print_every == 0:
                t_prints = t_prints.mean(dim=1)
                e_prints = e_prints.mean(dim=1)

                if not _g.args.quiet:
                    _u.pretty_print(_g.args.epochs, _g.args.time, epoch+1, epoch_end_time - start_time,
                                    _u.to_builtin(epoch_times.mean()), _u.to_builtin(torch.chunk(t_prints, 2)),
                                    _u.to_builtin(
                                        {n: (x, y) for n,x,y in
                                         zip(['loss'] + list(measures.keys()), e_prints[::2], e_prints[1::2])}
                                    )
                             )
                t_prints, e_prints = None, None

            if not best_eval_loss or e_losses[0].item() < best_eval_loss:
                saving = True
                best_eval_loss = e_losses[0].item()
                torch.save(model, _g.args.path + '/' + _g.args.filename)
                saving = False

            scheduler.step()

        if not _g.args.quiet:
            print('Training done successfully')

    except KeyboardInterrupt:
        print('\nExiting earlier than expected. Wait a moment!')

        if saving:  # In case it was interrupted while saving
            torch.save(model, _g.args.path + '/' + _g.args.filename)


def do_test(model, data, measures):
    start_time = time.time()
    input_size = len(_g.vocab)

    if not _g.args.quiet:
        print('Testing...')

    criterion = nn.NLLLoss(ignore_index=_g.vocab.stoi[_g.padding_symbol])

    losses = None
    so_far = 0

    try:
        for i, batch in zip(range(len(data)), data):  # TODO necessary for now to do it this way
            loss = _t.evaluate(model, criterion, _u.to_one_hot(batch.before, input_size), batch.after, measures)
            loss = loss.unsqueeze(dim=1)
            losses = loss if losses is None else torch.cat((losses, loss), dim=1)
            so_far = i+1

        if not _g.args.quiet:
            print('Testing done successfully')

    except KeyboardInterrupt:
        print('\nExiting earlier than expected. Wait a moment!')

    losses = losses.mean(dim=1)
    text = 'Test {} elements in {}.'.format(so_far * data.batch_size, _u.pretty_print_time(time.time() - start_time))
    eval_measures = _u.to_builtin({n: (x,y) for n,x,y in
                                   zip(['loss'] + list(measures.keys()), losses[::2], losses[1::2])})

    for i, j in eval_measures.items():
        text += ' ' + i + ' {:5.6f}({:5.6f}).'.format(j[0], j[1])
    if not _g.args.quiet:
        print(text)


def do_infer(model, data, device):
    if _g.args.quiet:
        return

    print("Infering {} test examples...".format(_g.args.examples))

    for i in range(_g.args.examples):
        print("Case {}:".format(i+1))
        index = random.randrange(0, len(data.examples))
        sample = data.examples[index]

        if _g.args.debug:
            _i.predict_and_plot(model, sample.before, sample.after, device)
        else:
            _i.predict(model, sample.before, sample.after, device)


def do_real_input(model, device):
    if _g.args.quiet:
        return

    for i in range(_g.args.examples):
        print("Case {}:".format(i+1))

        before = list(input('Input text: '))
        after = input('Target? (y/n): ')
        while after != "y" and after != "n":
            after = input('Target? (y/n): ')

        if after == "y":
            after = list(input('Target text: '))
            if _g.args.debug:
                _i.predict_and_plot(model, before, after, device)
            else:
                _i.predict(model, before, after, device)
        else:
            _i.predict(model, before, device=device)


if __name__ == '__main__':
    if _g.args.toy:
       _g.args.k1 = 3
       _g.args.k2 = 4
       _g.args.debug = True
       _g.args.cuda = False
       _g.args.bs = 3
       _g.args.epochs = 10
       _g.args.train = 'test/output_16_r.csv'
       _g.args.val = 'test/output_16_r.csv'
       _g.args.test = 'test/output_16_r.csv'
       _g.args.filerows = 1000

    if not _g.args.quiet:
        print(_g.args)

    # For reproducibility
    if _g.args.seed is not None:
        torch.backends.cudnn.deterministic = True
        random.seed(_g.args.seed)
        torch.manual_seed(_g.args.seed)
        np.random.seed(_g.args.seed)
        if _g.args.cuda:
            torch.cuda.manual_seed_all(_g.args.seed)

    device = torch.device('cuda' if _g.args.cuda else 'cpu')

    measures = {
        'cer': None,
        'acc': Accuracy(reduce=False)
    }

    # Building the dataset
    field = torchdata.Field(tokenize=lambda s: list(s), init_token=_g.start_symbol, eos_token=_g.end_symbol,
                            sequential=True, batch_first=True, pad_token=_g.padding_symbol, use_vocab=True)

    train, val, test = torchdata.TabularDataset.splits(
        path='', train=_g.args.train, validation=_g.args.val, test=_g.args.test, skip_header=False,
        format='csv', fields=(('before', field), ('after', field))
    )

    tdata, vdata, tedata = torchdata.BucketIterator.splits((train, val, test), batch_sizes=[_g.args.bs] * 3, sort=False,
                                                           repeat=True, sort_within_batch=False, shuffle=True,
                                                           sort_key=lambda x: len(x.before),
                                                           device=(-1 if not _g.args.cuda else None))

    model = None

    if not _g.args.load:
        field.build_vocab(train, val, test)
        _g.vocab = field.vocab
        measures['cer'] = CERLoss(_g.vocab.stoi[_g.padding_symbol], reduce=False)

        model = TextNormalizer(field.vocab, _g.args.nf, _g.args.ks, _g.args.rf, _g.args.attn, _g.args.hidden,
                               _g.args.midlayer, _g.args.dropout, _g.args.encoder).to(device)
        do_train(model, tdata, vdata, {}) #, measures)

    if _g.args.examples or not _g.args.notest:
        model = torch.load(_g.args.path + '/' + _g.args.filename, map_location='cpu').to(device)
        field.vocab = model.vocab
        _g.vocab = model.vocab
        measures['cer'] = CERLoss(_g.vocab.stoi[_g.padding_symbol], reduce=False)

    if not _g.args.notest:
        do_test(model, tedata, measures)

    if _g.args.examples > 0:
        if _g.args.input:
            do_real_input(model, device)
        else:
            do_infer(model, test, device)

    exit(0)


