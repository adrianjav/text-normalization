import random
import torch
import globals as _g
from utils import add_gradient, repackage_variable, to_one_hot


def tbtt(model, criterion, optimizer, data, target):
    def optim_step():
        optimizer.zero_grad()

        for j in range(1, len(states)):
            states[-j][2].backward(retain_graph=True)
            add_gradient(states[-j-1][1], states[-j][0])  # transfer gradients of same variable

        torch.nn.utils.clip_grad_norm(model.decoder.parameters(), _g.args.clip, 'inf')
        optimizer.step()

    target_sizes = ((target != _g.vocab.stoi[_g.end_symbol]).sum(1) - 1).float()

    with model.train(data, target.size()[1]) as seq:
        total_loss = torch.zeros_like(target_sizes, dtype=torch.float)
        state = (seq.inp, seq.hidden, seq.cell)
        states = [(None, state, None)]

        for i, out in enumerate(seq):
            loss = criterion(out, target[:, i + 1])
            total_loss += loss.detach()

            states.append((state, (seq.inp, seq.hidden, seq.cell), loss.mean()))
            if len(states) > _g.args.k2:
                del states[0]

            seq.inp, seq.hidden, seq.cell = repackage_variable(seq.inp, seq.hidden, seq.cell)
            state = (seq.inp, seq.hidden, seq.cell)

            if random.random() < _g.args.teacher:
                seq.inp = to_one_hot(target[:, i + 1], len(model.vocab)).squeeze(2) * 100 - 100

            if (i + 1) % _g.args.k1 == 0:
                optim_step()

        if len(seq) % _g.args.k1 != 0:
            optim_step()

        error = (total_loss / target_sizes)
        return torch.stack((error.mean(), error.std()), dim=0)


def evaluate(model, criterion, data, target, measures=None):
    with torch.no_grad():
        with model.eval(data, target.size()[1]) as seq:
            target_sizes = ((target != 1).sum(1) - 1).float()
            prediction = [seq.inp]
            total_loss = torch.zeros_like(target_sizes)

            for i, out in enumerate(seq):
                prediction.append(seq.inp)
                total_loss += criterion(seq.inp, target[:, i + 1])

            error = (total_loss / target_sizes)
            results = [error.mean(), error.std()]

            if measures is not None:
                prediction = torch.stack(prediction[1:], dim=1)
                for f in measures.values():
                    r = f(prediction, target[..., 1:]) / target_sizes
                    results += [r.mean(), r.std()]

            return torch.stack(results, dim=0)

