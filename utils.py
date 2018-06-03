import random
import numpy as np
import matplotlib.pyplot as plt


def one_hot_conversion(s, char_steps, alphabet):
    """
    create one hot vectors from sentences
    """
    steplimit = 3e3
    s = s[:3e3] if len(s) > 3e3 else s
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= char_steps:
        seq = seq[:char_steps]
    else:
        seq = seq + [0] * (char_steps - len(seq))
    one_hot = np.zeros((char_steps, len(alphabet) + 1))
    one_hot[np.arange(char_steps), seq] = 1
    return one_hot


def batches(batch_size, data, labels, sentences, char_steps, alphabat):
    '''
    Return random samples and labels. 
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    sentences_suffle = [sentences[i] for i in idx]
    one_hots = [
        one_hot_conversion(s, char_steps, alphabat) for s in sentences_suffle
    ]
    return data_shuffle, labels_shuffle, sentences_suffle, one_hots


def plot_strokes(strokes, title, figsize=(20, 2)):
    """
    plot strokes 
    """
    
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:, -1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1]
    for i in range(len(eos_preds) - 1):
        start = eos_preds[i] + 1
        stop = eos_preds[i + 1]
        plt.plot(
            strokes[start:stop, 1],
            strokes[start:stop, 0],
            'k-',
            linewidth=2.0)
    plt.title(title, fontsize=20)
    plt.show()

def plot_random_stroke(stroke, save_name=None):
    # Plot a single example.
    i=[2,0,1]
    stroke=stroke[:,i]
    f, ax = plt.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        plt.show()
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print "Error building image!: " + save_name

    plt.close()