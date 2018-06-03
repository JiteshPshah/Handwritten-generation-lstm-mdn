import tensorflow as tf
import numpy as np
import cPickle as pickle
import time
from utils import batches
import os
import sys


class build_model():
    """
    design a model to generate random handwritting 
    """
    def __init__(self):
        self.tsteps = 300
        self.bias = 1.0
        self.lstm_units = 128
        self.mixture_comps = 1
        self.n_gaussian_mixtures = 8
        self.grad_clip = 10.
        self.rnn_size = 100
        self.batch_size = 32
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

    def load_checkpoints(self, save_path):
        global_step = 0
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, load_path)
        print("checkpoint model is loaded: {}".format(load_path))
        self.saver = tf.train.Saver(tf.global_variables())
        global_step = int(load_path.split('-')[-1])
        return global_step


decay_value = 0.95
momentum_value = 0.9
learning_rate = 1e-4
lr_decay = 1.0
save_path = sys.argv[1]
if not os.path.exists(save_path):
    os.mkdir(save_path)
no_batches = 500
epoches = 250
global_step = 0
save_batches = 500
alphabets = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

model = build_model()

if os.listdir(save_path) != []:
    print "chekpoint model loaded"
    global_step = model.load_checkpoints(save_path+"model.ckpt")
[training_data, Y, sentences, one_hot_vectors] = pickle.load(
    open(sys.argv[2]))
input_data_dict = {
    model.input_data: training_data,
    model.target_Data: Y,
    model.state_in: model.state_in.eval()
}
state = model.state_in.eval()
model.sess.run(tf.assign(model.decay, decay_value))
model.sess.run(tf.assign(model.momentum, momentum_value))
tsteps_per_characters = 25
char_steps = model.tsteps / tsteps_per_characters

for i in range(global_step / no_batches, epoches):
    print i
    model.sess.run(tf.assign(model.lr, learning_rate * lr_decay**i))
    print "learning_rate:{}".format(model.lr.eval())
    c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(
    ), model.istate_cell2.c.eval()
    h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(
    ), model.istate_cell2.h.eval()
    for b in range(global_step % no_batches, no_batches):
        a = i * no_batches + b
        if global_step is not 0:
            a += 1
            global_step = 0
        if a % save_batches == 0 and (a > 0):
            model.saver.save(model.sess, save_path, global_step=a)

        x, y, s, c = batches(model.batch_size, training_data, Y, sentences,
                             char_steps, alphabets)
        my_feed_dict = {
            model.input_data: x,
            model.target_Data: y,
            model.state_in: state,
            model.istate_cell0.c: c0,
            model.istate_cell1.c: c1,
            model.istate_cell2.c: c2,
            model.istate_cell0.h: h0,
            model.istate_cell1.h: h1,
            model.istate_cell2.h: h2
        }

        [training_loss, state, _] = model.sess.run(
            [model.cost, model.state_out, model.train_ops], my_feed_dict)
        my_feed_dict.update(input_data_dict)

        if a % 10 is 0:
            print("{}/{}, loss = {:.4f}".format(a, epoches * no_batches,
                                                training_loss))