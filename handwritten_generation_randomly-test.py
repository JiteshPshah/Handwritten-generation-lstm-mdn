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
    """
    initiliaze a model to genrate handwriting 
    """
    def __init__(self, save_path):
        self.tsteps = 1
        self.bias = 1.0
        self.lstm_units = 128
        self.mixture_comps = 1
        self.n_gaussian_mixtures = 8
        self.grad_clip = 10.
        self.rnn_size = 100
        self.batch_size = 1
        self.dropout = 0.85

        self.graves_initializer = tf.truncated_normal_initializer(
            mean=0., stddev=.075, seed=None, dtype=tf.float32)
        self.w_b_initializer = tf.truncated_normal_initializer(
            mean=-3.0, stddev=.25, dtype=tf.float32, seed=None)

        self.cell0 = tf.contrib.rnn.LSTMCell(
            self.rnn_size,
            state_is_tuple=True,
            initializer=self.graves_initializer)
        self.cell1 = tf.contrib.rnn.LSTMCell(
            self.rnn_size,
            state_is_tuple=True,
            initializer=self.graves_initializer)
        self.cell2 = tf.contrib.rnn.LSTMCell(
            self.rnn_size,
            state_is_tuple=True,
            initializer=self.graves_initializer)

        # if (self.train and self.dropout < 1): # training mode
        self.cell0 = tf.contrib.rnn.DropoutWrapper(
            self.cell0, output_keep_prob=self.dropout)
        self.cell1 = tf.contrib.rnn.DropoutWrapper(
            self.cell1, output_keep_prob=self.dropout)
        self.cell2 = tf.contrib.rnn.DropoutWrapper(
            self.cell2, output_keep_prob=self.dropout)

        self.input_data = tf.placeholder(
            dtype=tf.float32, shape=[None, self.tsteps, 3])
        self.target_Data = tf.placeholder(
            dtype=tf.float32, shape=[None, self.tsteps, 3])

        self.istate_cell0 = self.cell0.zero_state(
            batch_size=self.batch_size, dtype=tf.float32)
        self.istate_cell1 = self.cell1.zero_state(
            batch_size=self.batch_size, dtype=tf.float32)
        self.istate_cell2 = self.cell2.zero_state(
            batch_size=self.batch_size, dtype=tf.float32)

        self.inputs = [
            tf.squeeze(i, [1])
            for i in tf.split(self.input_data, self.tsteps, 1)
        ]

        outs_cell0, self.fstate_cell0 = tf.contrib.legacy_seq2seq.rnn_decoder(
            self.inputs,
            self.istate_cell0,
            self.cell0,
            loop_function=None,
            scope='cell0')

        outs_cell1, self.fstate_cell1 = tf.contrib.legacy_seq2seq.rnn_decoder(
            outs_cell0,
            self.istate_cell1,
            self.cell1,
            loop_function=None,
            scope='cell1')

        outs_cell2, self.fstate_cell2 = tf.contrib.legacy_seq2seq.rnn_decoder(
            outs_cell1,
            self.istate_cell2,
            self.cell2,
            loop_function=None,
            scope='cell2')

        # build Mixture density network
        n_out = 1 + self.n_gaussian_mixtures * 6
        with tf.variable_scope("mdn_dense"):
            mdn_w = tf.get_variable(
                "output_w", [self.rnn_size, n_out],
                initializer=self.graves_initializer)
            mdn_b = tf.get_variable(
                "output_b", [n_out], initializer=self.graves_initializer)
        out_cell2 = tf.reshape(tf.concat(outs_cell2, 1), [-1, self.rnn_size])
        self.output = tf.nn.xw_plus_b(out_cell2, mdn_w, mdn_b)
        self.state_in = tf.identity(self.istate_cell2, name='state_in')
        self.state_out = tf.identity(self.fstate_cell2, name='state_out')

        def gaussian(x1, x2, mu1, mu2, sig1, sig2, rho):
            xmu1 = tf.subtract(x1, mu1)
            xmu2 = tf.subtract(x2, mu2)
            Z = tf.square(tf.div(xmu1, sig1)) + tf.square(tf.div(
                xmu2, sig2)) - 2 * tf.div(
                    tf.multiply(rho, tf.multiply(xmu1, xmu2)),
                    tf.multiply(sig1, sig2))
            rho_sqr = 1 - tf.square(rho)
            numerator = tf.exp(tf.div(-Z, 2 * rho_sqr))
            denominator = 2*np.pi * \
                tf.multiply(tf.multiply(sig1, sig2), tf.sqrt(rho_sqr))
            N = tf.div(numerator, denominator)
            return N

        def define_loss(pi, x1, x2, eos_data, mu1, mu2, sig1, sig2, rho, eos):
            N = gaussian(x1, x2, mu1, mu2, sig1, sig2, rho)
            first_term = tf.multiply(N, pi)
            summation = tf.reduce_sum(first_term, 1, keep_dims=True)
            log_sum = -tf.log(tf.maximum(summation, 1e-20))
            second_term = tf.multiply(eos, eos_data) + \
                tf.multiply(1-eos, 1-eos_data)
            log_of_second_term = -tf.log(second_term)
            total_sum = tf.reduce_sum(log_sum + log_of_second_term)
            return total_sum

        def mdn_parameters(Z):
            eos_hat = Z[:, 0:1]
            pi_hat, mu1_hat, mu2_hat, sig1_hat, sig2_hat, rho_hat = tf.split(
                Z[:, 1:], 6, 1)
            self.pi_hat, self.sig1_hat, self.sig2_hat = pi_hat, sig1_hat, sig2_hat
            eos = tf.sigmoid(-1 * eos_hat)
            pi = tf.nn.softmax(pi_hat)
            mu1 = mu1_hat
            mu2 = mu2_hat
            sig1 = tf.exp(sig1_hat)
            sig2 = tf.exp(sig2_hat)
            rho = tf.tanh(rho_hat)
            return [eos, pi, mu1, mu2, sig1, sig2, rho]

        flat_target_data = tf.reshape(self.target_Data, [-1, 3])

        [x1_data, x2_data, eos_data] = tf.split(flat_target_data, 3, 1)

        [
            self.eos, self.pi, self.mu1, self.mu2, self.sig1, self.sig2,
            self.rho
        ] = mdn_parameters(self.output)

        seq_loss = define_loss(self.pi, x1_data, x2_data, eos_data, self.mu1,
                               self.mu2, self.sig1, self.sig2, self.rho,
                               self.eos)
        self.cost = seq_loss / (self.batch_size * self.tsteps)

        self.lr = tf.Variable(0.0, trainable=False)
        self.decay = tf.Variable(0.0, trainable=False)
        self.momentum = tf.Variable(0.0, trainable=False)

        self.trainable_vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, self.trainable_vars), self.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.lr, decay=self.decay, momentum=self.momentum)

        self.train_ops = optimizer.apply_gradients(
            zip(self.grads, self.trainable_vars))

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        global_step = 0
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, load_path)
        print("checkpoint model is loaded: {}".format(load_path))
        self.saver = tf.train.Saver(tf.global_variables())
        global_step = int(load_path.split('-')[-1])


save_path = sys.argv[1] + "model.ckpt"
model = build_model(save_path)


def test_gaussian(mu1, mu2, sig1, sig2, rho):
    mean = [mu1, mu2]
    cov = [[sig1 * sig1, rho * sig1 * sig2], [rho * sig1 * sig2, sig2 * sig2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    print('error with sampling ensemble')


def generate(model, random_seed=1, num=800):
    """
    this function returns a strokes given a random seed
    num is a number of timesteps
    the strokes dimension should be (num,3)
    """
    np.random.seed(random_seed)
    previous_x = np.asarray([[[0, 0, 1]]], dtype=np.float32)
    previous_x[0, 0, 2] = 1
    prev_state = model.sess.run(model.state_in)
    strokes = np.zeros((num, 3), dtype=np.float32)
    mixture_params = []
    for i in range(num):
        my_feed_dict = {
            model.input_data: previous_x,
            model.state_in: prev_state
        }
        fetch_list = [
            model.pi, model.mu1, model.mu2, model.sig1, model.sig2, model.rho,
            model.eos, model.state_out
        ]
        [pi, mu1, mu2, sig1, sig2, rho, eos_op, next_state] = model.sess.run(
            fetch_list, my_feed_dict)
        idx = get_pi_idx(random.random(), pi[0])
        eos_prob = 1 if random.random() < eos_op[0][0] else 0
        x1, x2 = test_gaussian(mu1[0][idx], mu2[0][idx], sig1[0][idx],
                               sig2[0][idx], rho[0][idx])
        strokes[i, :] = [x1, x2, eos_prob]
        previous_x = np.zeros((1, 1, 3), dtype=np.float32)
        previous_x[0][0] = np.array(
            [mu1[0][idx], mu2[0][idx], eos_prob], dtype=np.float32)
        prev_state = next_state
    strokes[:, 0:2] *= 20
    return strokes


random_seed = int(sys.argv[2])
timesteps=sys.argv[3]
file_name = sys.argv[4]
strokes = generate(model, random_seed)
np.save(str(file_name), strokes)