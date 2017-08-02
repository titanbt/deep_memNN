from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf
from sklearn import metrics
import lasagne.nonlinearities as nonlinearities

class SentenceMemNN(object):
    def __init__(self, nwords=0, init_hid=0.1, init_std=0.05, batch_size=128, num_epochs=100, num_hops=7, embedd_dim=300,
                 mem_size=78, lindim=75, max_grad_norm=50, pad_idx=0, embedd_table_aspect=None, lr=0.01):

        self.nwords = nwords
        self.init_hid = init_hid
        self.init_std = init_std
        self.batch_size = batch_size
        self.nepoch = num_epochs
        self.nhop = num_hops
        self.edim = embedd_dim
        self.mem_size = mem_size
        self.lindim = lindim
        self.max_grad_norm = max_grad_norm
        self.pad_idx = pad_idx
        self.embedd_table_aspect = embedd_table_aspect
        self.lr = lr
        self.hid = []

        self.aspect = tf.placeholder(tf.int32, [self.batch_size, 1], name="input")
        self.loc = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="location")
        self.target = tf.placeholder(tf.float32, [self.batch_size, 3], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")


    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.ASP = tf.Variable(tf.random_normal([self.embedd_table_aspect.shape[0], self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))
        self.BL_W = tf.Variable(tf.random_normal([2 * self.edim, 1], stddev=self.init_std))
        self.BL_B = tf.Variable(tf.zeros([1, 1]))

        # Location Encoding
        self.T_A = tf.Variable(tf.random_normal([self.mem_size + 1, self.edim], stddev=self.init_std))
        self.T_B = tf.Variable(tf.random_normal([self.mem_size + 1, self.edim], stddev=self.init_std))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.loc)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.loc)
        Bin = tf.add(Bin_c, Bin_t)

        ASPin = tf.nn.embedding_lookup(self.ASP, self.aspect)
        ASPout2dim = tf.reshape(ASPin, [-1, self.edim])
        self.hid.append(ASPout2dim)

        for h in xrange(self.nhop):
            til_hid = tf.tile(self.hid[-1], [1, self.mem_size])
            til_hid3dim = tf.reshape(til_hid, [-1, self.mem_size, self.edim])
            a_til_concat = tf.concat([til_hid3dim, Ain], 2)
            til_bl_wt = tf.tile(self.BL_W, [self.batch_size, 1])
            til_bl_3dim = tf.reshape(til_bl_wt, [self.batch_size, -1, 2 * self.edim])
            att = tf.matmul(a_til_concat, til_bl_3dim, adjoint_b=True)
            til_bl_b = tf.tile(self.BL_B, [self.batch_size, self.mem_size])
            til_bl_3dim = tf.reshape(til_bl_b, [-1, self.mem_size, 1])
            g = tf.nn.tanh(tf.add(att, til_bl_3dim))
            g_2dim = tf.reshape(g, [-1, self.mem_size])
            P = tf.nn.softmax(g_2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], self.C)
            Dout = tf.add(Cout, Bout2dim)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim - self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat([F, K], 1))


    def build_network(self):

        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, 3], stddev=self.init_std))
        z = tf.matmul(self.hid[-1], self.W)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.target)

        self.lr = tf.Variable(self.lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W, self.ASP, self.BL_W, self.BL_B]
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.initialize_all_variables().run()

        self.correct_prediction = tf.argmax(z, 1)
