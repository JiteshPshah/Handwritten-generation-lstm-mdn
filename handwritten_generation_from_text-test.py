import numpy as np
import numpy.matlib
import math
import random
import time
import os
import cPickle as pickle
from utils import batches, one_hot_conversion
import tensorflow as tf
import sys

class build_model():
    def __init__(self):
        pass


model = build_model()

save_path = sys.argv[1]+"model.ckpt"
text = sys.argv[2]
tsteps = 1
mixture_comps = 1
tsteps_per_characters = 25
model.char_steps = len(text)
n_gaussian_mixtures = 8
alphabets = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
model.char_seqeunce_len = len(alphabets) + 1
rnn_size = 100
batch_size = 1

model.graves_initializer = tf.truncated_normal_initializer(
    mean=0., stddev=.075, seed=None, dtype=tf.float32)
model.w_b_initializer = tf.truncated_normal_initializer(
    mean=-3.0, stddev=.25, dtype=tf.float32, seed=None)

cell_func = tf.contrib.rnn.LSTMCell  # could be GRUCell or RNNCell
model.cell0 = cell_func(
    rnn_size, state_is_tuple=True, initializer=model.graves_initializer)
model.cell1 = cell_func(
    rnn_size, state_is_tuple=True, initializer=model.graves_initializer)
model.cell2 = cell_func(
    rnn_size, state_is_tuple=True, initializer=model.graves_initializer)

model.input_data = tf.placeholder(dtype=tf.float32, shape=[None, tsteps, 3])
model.target_Data = tf.placeholder(dtype=tf.float32, shape=[None, tsteps, 3])

model.istate_cell0 = model.cell0.zero_state(
    batch_size=batch_size, dtype=tf.float32)
model.state_in = tf.identity(model.istate_cell0, name='state_in')
model.istate_cell1 = model.cell1.zero_state(
    batch_size=batch_size, dtype=tf.float32)
model.istate_cell2 = model.cell2.zero_state(
    batch_size=batch_size, dtype=tf.float32)
inputs = [tf.squeeze(i, [1]) for i in tf.split(model.input_data, tsteps, 1)]

outs_cell0, model.fstate_cell0 = tf.contrib.legacy_seq2seq.rnn_decoder(
    inputs, model.istate_cell0, model.cell0, loop_function=None, scope='cell0')

# make a discrete convolution soft window for handwritten synthesis

def create_soft_window(alpha, beta, kappa, cu):
    char_steps = cu.get_shape()[1].value
    phi_tu = discrete_convolution(char_steps, alpha, beta, kappa)
    wt = tf.matmul(phi_tu, cu)
    wt = tf.squeeze(wt, [1])
    return wt, phi_tu


def discrete_convolution(char_steps, alpha, beta, kappa):
    u = np.linspace(0, char_steps - 1, char_steps)
    kappa_1 = tf.square(tf.subtract(kappa, u))
    expon = tf.multiply(-beta, kappa_1)
    phi_1 = tf.multiply(alpha, tf.exp(expon))
    phi_tu = tf.reduce_sum(phi_1, 1, keepdims=True)
    return phi_tu


def window_parameters(i, out_cell0, mixture_comps, previous_kappa, reuse=True):
    hidden = out_cell0.get_shape()[1]
    n_out = 3 * mixture_comps

    with tf.variable_scope('window', reuse=reuse):
        window_w = tf.get_variable(
            "window_w", [hidden, n_out], initializer=model.graves_initializer)
        window_b = tf.get_variable(
            "window_b", [n_out], initializer=model.w_b_initializer)
    alph_beta_kapa_hat = tf.nn.xw_plus_b(out_cell0, window_w, window_b)
    alph_beta_kapa = tf.exp(
        tf.reshape(alph_beta_kapa_hat, [-1, 3 * mixture_comps, 1]))

    alpha, beta, kappa = tf.split(alph_beta_kapa, 3, 1)
    kappa = kappa + previous_kappa
    return alpha, beta, kappa


model.init_kappa = tf.placeholder(
    dtype=tf.float32, shape=[None, mixture_comps, 1])
model.char_sequence = tf.placeholder(
    dtype=tf.float32, shape=[None, model.char_steps, model.char_seqeunce_len])
avg_prev_kappa = model.init_kappa
prev_w = model.char_sequence[:, 0, :]

# combine hidden layer 1 with attension mechanism
reuse = False
for i in range(len(outs_cell0)):
    [alpha, beta, updated_kappa] = window_parameters(
        i, outs_cell0[i], mixture_comps, avg_prev_kappa, reuse=reuse)
    window, phi = create_soft_window(alpha, beta, updated_kappa,
                                     model.char_sequence)
    outs_cell0[i] = tf.concat((outs_cell0[i], window), 1)
    outs_cell0[i] = tf.concat((outs_cell0[i], inputs[i]), 1)
    avg_prev_kappa = tf.reduce_mean(
        updated_kappa, reduction_indices=1, keepdims=True)
    reuse = True

model.window = window
model.phi = phi
model.updated_kappa = updated_kappa
model.alpha = alpha
model.avg_prev_kappa = avg_prev_kappa

outs_cell1, model.fstate_cell1 = tf.contrib.legacy_seq2seq.rnn_decoder(
    outs_cell0,
    model.istate_cell1,
    model.cell1,
    loop_function=None,
    scope='cell1')

outs_cell2, model.fstate_cell2 = tf.contrib.legacy_seq2seq.rnn_decoder(
    outs_cell1,
    model.istate_cell2,
    model.cell2,
    loop_function=None,
    scope='cell2')

out_cell2 = tf.reshape(tf.concat(outs_cell2, 1), [-1, rnn_size])
# build Mixture density network
n_out = 1 + n_gaussian_mixtures * 6
with tf.variable_scope("mdn_dense"):
    output_w = tf.get_variable(
        "output_w", [rnn_size, n_out], initializer=model.graves_initializer)
    output_b = tf.get_variable(
        "output_b", [n_out], initializer=model.graves_initializer)

output = tf.nn.xw_plus_b(out_cell2, output_w, output_b)
model.state_out = tf.identity(model.fstate_cell0, name='state_out')


def gaussian(x1, x2, mu1, mu2, sig1, sig2, rho):
    xmu1 = tf.subtract(x1, mu1)
    xmu2 = tf.subtract(x2, mu2)
    Z = tf.square(tf.div(xmu1, sig1)) + tf.square(tf.div(
        xmu2, sig2)) - 2 * tf.div(
            tf.multiply(rho, tf.multiply(xmu1, xmu2)), tf.multiply(sig1, sig2))
    rho_sqr = 1 - tf.square(rho)
    numerator = tf.exp(tf.div(-Z, 2 * rho_sqr))
    denominator = 2*np.pi * \
        tf.multiply(tf.multiply(sig1, sig2), tf.sqrt(rho_sqr))
    N = tf.div(numerator, denominator)
    return N


def mdn_parameters(Z):
    eos_hat = Z[:, 0:1]
    pi_hat, mu1_hat, mu2_hat, sig1_hat, sig2_hat, rho_hat = tf.split(
        Z[:, 1:], 6, 1)
    model.pi_hat, model.sig1_hat, model.sig2_hat = pi_hat, sig1_hat, sig2_hat
    eos = tf.sigmoid(-1 * eos_hat)
    pi = tf.nn.softmax(pi_hat)
    mu1 = mu1_hat
    mu2 = mu2_hat
    sig1 = tf.exp(sig1_hat)
    sig2 = tf.exp(sig2_hat)
    rho = tf.tanh(rho_hat)
    return [eos, pi, mu1, mu2, sig1, sig2, rho]


flat_target_data = tf.reshape(model.target_Data, [-1, 3])

[x1_data, x2_data, eos_data] = tf.split(flat_target_data, 3, 1)

[model.eos, model.pi, model.mu1, model.mu2, model.sig1, model.sig2,
 model.rho] = mdn_parameters(output)

model.sess = tf.InteractiveSession()
model.saver = tf.train.Saver(tf.global_variables())
model.sess.run(tf.global_variables_initializer())

global_step = 0
save_dir = '/'.join(save_path.split('/')[:-1])
ckpt = tf.train.get_checkpoint_state(save_dir)
load_path = ckpt.model_checkpoint_path
model.saver.restore(model.sess, load_path)
print "loaded model: {}".format(load_path)
model.saver = tf.train.Saver(tf.global_variables())
global_step = int(load_path.split('-')[-1])


def test_gaussian(mu1, mu2, sig1, sig2, rho):
    mean = [mu1, mu2]
    cov = [[sig1 * sig1, rho * sig1 * sig2], [rho * sig1 * sig2, sig2 * sig2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def test(text, model, bias, random_seed=1):
    """
    given a text string this function returns a stroke values
    """
    np.random.seed(random_seed)
    one_hot_vector = [one_hot_conversion(text, model.char_steps, alphabets)]
    c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(
    ), model.istate_cell2.c.eval()
    h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(
    ), model.istate_cell2.h.eval()
    kappa = np.zeros((1, mixture_comps, 1))
    previous_x = np.asarray([[[0, 0, 1]]], dtype=np.float32)
    strokes, pis, windows, phis, kappas = [], [], [], [], []
    terminate = False
    i = 0
    while not terminate:
        my_feed_dict = {
            model.input_data: previous_x,
            model.char_sequence: one_hot_vector,
            model.init_kappa: kappa,
            model.istate_cell0.c: c0,
            model.istate_cell1.c: c1,
            model.istate_cell2.c: c2,
            model.istate_cell0.h: h0,
            model.istate_cell1.h: h1,
            model.istate_cell2.h: h2
        }

        fetch_list = [
            model.pi_hat, model.mu1, model.mu2, model.sig1_hat, model.sig2_hat,
            model.rho, model.eos, model.window, model.phi, model.updated_kappa,
            model.avg_prev_kappa, model.alpha, model.fstate_cell0.c,
            model.fstate_cell1.c, model.fstate_cell2.c, model.fstate_cell0.h,
            model.fstate_cell1.h, model.fstate_cell2.h
        ]

        [
            pi_hat, mu1, mu2, sig1_hat, sig2_hat, rho, eos, window, phi, kappa,
            avg_kappa, alpha, c0, c1, c2, h0, h1, h2
        ] = model.sess.run(fetch_list, my_feed_dict)

        sig1 = np.exp(sig1_hat - bias)
        sig2 = np.exp(sig2_hat - bias)
        pi_hat *= 1 + bias
        pi = np.zeros_like(pi_hat)
        pi[0] = np.exp(pi_hat[0]) / np.sum(np.exp(pi_hat[0]), axis=0)
        idx = np.random.choice(pi.shape[1], p=pi[0])
        eos = 1 if 0.5 < eos[0][0] else 0
        x1, x2 = test_gaussian(mu1[0][idx], mu2[0][idx], sig1[0][idx],
                               sig2[0][idx], rho[0][idx])
        windows.append(window)
        phis.append(phi[0])
        kappas.append(kappa[0])
        pis.append(pi[0])
        strokes.append([mu1[0][idx], mu2[0][idx], eos])
        kappa_idx = np.where(alpha[0] == np.max(alpha[0]))

        terminate = True if kappa[0][kappa_idx] > len(text) + 1 else False

        previous_x[0][0] = np.array([x1, x2, eos], dtype=np.float32)
        kappa = avg_kappa
        i += 1

    windows = np.vstack(windows)
    phis = np.vstack(phis)
    kappas = np.vstack(kappas)
    pis = np.vstack(pis)
    strokes = np.vstack(strokes)
    strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
    return strokes


bias = 1.0
random_seed=sys.argv[3]
strokes = test(text, model, bias, random_seed=1)
output_data={}
output_data['strokes']=strokes
output_data['text']=text
with open(str(sys.argv[4]),'w') as fp:
    pickle.dump(output_data,fp)