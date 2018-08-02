import torch
import numpy as np
import matplotlib.pyplot as plt
import globals as _g
from utils import to_one_hot


def text_to_input(text):
    inp = [_g.vocab.stoi[_g.start_symbol]] + [_g.vocab.stoi[x] for x in text] + [_g.vocab.stoi[_g.end_symbol]]
    inp = torch.tensor(inp).unsqueeze(0)
    return to_one_hot(inp, len(_g.vocab))


def output_to_text(output):
    text = torch.max(output, dim=1)[1].cpu().numpy()
    text = [_g.vocab.itos[x] for x in text]
    return ''.join(text)


def predict(model, text, target = None, device=torch.device('cpu')):
    with torch.no_grad():
        inp = text_to_input(text).to(device)
        with model.eval(inp) as seq:
            prediction = [seq.inp]
            for out in seq:
                prediction += [out]

        out = output_to_text(torch.cat(prediction[1:-1], dim=0))
        print_results(text, target, out)

        return out


def predict_and_plot(model, text, target, device=torch.device('cpu')):
    with torch.no_grad():
        inp = text_to_input(text).to(device)

        with model.eval(inp) as seq:
            prediction, attn = [seq.inp], []
            for out in seq:
                attn += [seq.attn]
                prediction += [out]

        prediction = torch.cat(prediction[1:], dim=0)
        attn = torch.cat(attn[:-1], dim=0)
        out = output_to_text(prediction[:-1])

        print_results(text, target, out)
        print_plausible_letters(target, prediction)
        plot_heatmap(text, out, attn, min(len(target), attn.size()[0]))

        return out


def print_results(text, target, prediction):
    print("Input:      {}".format(''.join(text)))
    if target is not None:
        print("Output:     {}".format(''.join(target)))
    print("Prediction: {}".format(prediction))
    print("")


def plot_heatmap(word, prediction, values, size):
    # attn is a tensor of size [prediction size, word size]
    fig, axis = plt.subplots(figsize=(size, len(word)))
    axis.imshow(values[:size].cpu().numpy(), cmap=plt.cm.Reds, interpolation='nearest')

    axis.set_xticks(range(len(word)))
    axis.set_xticklabels(word)
    axis.xaxis.tick_top()

    axis.set_yticks(range(size))
    axis.set_yticklabels(prediction[:size])

    axis.set_aspect('auto')
    plt.show()


def print_plausible_letters(target, probs):
    target = target + [_g.end_symbol]
    print(" target  |  first  |  second |  third  ")
    probs = torch.exp(probs)
    probs, indexes = torch.sort(probs, dim=1, descending=True)

    probs = probs.data.cpu().numpy()
    indexes = indexes.data.cpu().numpy()

    for i, c in zip(range(probs.shape[0]), target):
        j, = np.where(indexes[i,] == _g.vocab.stoi[c])
        j = j[0]

        print(" {} {:1.3f} | {} {:1.3f} | {} {:1.3f} | {} {:1.3f} ".format(c, probs[i, j],
                                                                           _g.vocab.itos[indexes[i, 0]], probs[i, 0],
                                                                           _g.vocab.itos[indexes[i, 1]], probs[i, 1],
                                                                           _g.vocab.itos[indexes[i, 2]], probs[i, 2]
                                                                           ))
    if probs.shape[0] < len(target):
        print('And the final part "{}" is missing'.format(target[probs.shape[0]:]))
    elif probs.shape[0] > len(target):
        print('And it has {} extra characters'.format(probs.shape[0] - len(target)))
    print("")
