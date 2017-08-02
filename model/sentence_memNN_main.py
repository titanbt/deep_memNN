from ConfigParser import SafeConfigParser
from loader.data_helper import DataHelper
from nn.sentence_memNN import SentenceMemNN
import numpy as np
import math
import tensorflow as tf

class SentenceMemNNMain(object):

    def __init__(self, config=None, opts=None):
        if config == None and opts == None:
            print "Please specify command option or config file ..."
            return

        parser = SafeConfigParser()
        parser.read(config)

        self.train_file = parser.get('model', 'train_file')
        self.test_file = parser.get('model', 'test_file')
        self.embedding_path = parser.get('model', 'embedding_path')

        self.batch_size = parser.getint('model', 'batch_size')
        self.num_epochs = parser.getint('model', 'num_epochs')

        self.lindim = parser.getint('model', 'lindim')
        self.init_hid = parser.getfloat('model', 'init_hid')
        self.init_std = parser.getfloat('model', 'init_std')
        self.embedd_dim = parser.getint('model', 'embedd_dim')
        self.max_grad_norm = parser.getint('model', 'max_grad_norm')
        self.num_hops = parser.getint('model', 'num_hops')

        self.lr = parser.getfloat('model', 'learning_rate')

        self.params = {'pad_idx': 0, 'nwords': 0, 'mem_size': 0, 'embedd_table': [], 'embedd_table_aspect': []}

        self.setupOperators()

    def setupOperators(self):
        print('Loading the training data...')
        self.reader = DataHelper(self.train_file,
                                 self.test_file,
                                 edim=self.embedd_dim,
                                 init_std=self.init_std,
                                 embedding_path=self.embedding_path)

        self.train_data, self.test_data, self.params = self.reader.load_data()

    def init_model(self):
        print "Building model..."
        self.model = SentenceMemNN(nwords=self.params['nwords'], init_hid=self.init_hid, init_std=self.init_std,
                                   batch_size=self.batch_size, num_epochs=self.num_epochs,
                                   num_hops=self.num_hops, embedd_dim=self.embedd_dim, mem_size=self.params['mem_size'],
                                   lindim=self.lindim, max_grad_norm=self.max_grad_norm, pad_idx=self.params['pad_idx'],
                                   embedd_table_aspect=self.params['embedd_table_aspect'], lr=self.lr)
        self.model.build_network()

    def run(self):
        with tf.Session() as sess:
            self.init_model()
            sess.run(self.model.A.assign(self.params['embedd_table']))
            sess.run(self.model.B.assign(self.params['embedd_table']))
            sess.run(self.model.ASP.assign(self.params['embedd_table_aspect']))
            for idx in xrange(self.num_epochs):
                print('epoch ' + str(idx) + '...')
                train_loss, train_acc = self.train(sess)
                test_loss, test_acc = self.test(sess, self.test_data)
                print('train-loss=%.2f;train-acc=%.2f;test-acc=%.2f;' % (train_loss, train_acc, test_acc))

    def train(self, sess):
        source_data, source_loc_data, aspect_data, aspect_label, _ = self.train_data
        N = int(math.ceil(len(source_data) / self.batch_size))
        cost = 0

        aspect = np.ndarray([self.batch_size, 1], dtype=np.float32)
        loc = np.ndarray([self.batch_size, self.params['mem_size']], dtype=np.int32)
        target = np.zeros([self.batch_size, 3])
        context = np.ndarray([self.batch_size, self.params['mem_size']])

        rand_idx, cur = np.random.permutation(len(source_data)), 0
        for idx in xrange(N):
            context.fill(self.params['pad_idx'])
            loc.fill(self.params['mem_size'])
            target.fill(0)

            self.set_variables(sess)

            for b in xrange(self.batch_size):
                m = rand_idx[cur]
                aspect[b][0] = aspect_data[m]
                target[b][aspect_label[m]] = 1
                loc[b, :len(source_loc_data[m])] = source_loc_data[m]
                context[b, :len(source_data[m])] = source_data[m]
                cur = cur + 1

            a, loss, self.step = sess.run([self.model.optim, self.model.loss, self.model.global_step],
                                          feed_dict={self.model.aspect: aspect,
                                                     self.model.loc: loc,
                                                     self.model.target: target,
                                                     self.model.context: context})
            cost += np.sum(loss)

        _, train_acc = self.test(sess, self.train_data)
        return cost / N / self.batch_size, train_acc

    def test(self, sess, data):
        source_data, source_loc_data, target_data, target_label, _ = data
        N = int(math.ceil(len(source_data) / self.batch_size))
        cost = 0

        aspect = np.ndarray([self.batch_size, 1], dtype=np.float32)
        loc = np.ndarray([self.batch_size, self.params['mem_size']], dtype=np.int32)
        target = np.zeros([self.batch_size, 3])
        context = np.ndarray([self.batch_size, self.params['mem_size']])
        context.fill(self.params['pad_idx'])

        self.set_variables(sess)

        m, acc = 0, 0
        for i in xrange(N):
            context.fill(self.params['pad_idx'])
            loc.fill(self.params['mem_size'])
            target.fill(0)

            raw_labels = []
            for b in xrange(self.batch_size):
                aspect[b][0] = target_data[m]
                target[b][target_label[m]] = 1
                loc[b, :len(source_loc_data[m])] = source_loc_data[m]
                context[b, :len(source_data[m])] = source_data[m]
                m += 1
                raw_labels.append(target_label[m])

            a, loss, self.step = sess.run([self.model.optim, self.model.loss, self.model.global_step],
                                           feed_dict={
                                               self.model.aspect: aspect,
                                               self.model.loc: loc,
                                               self.model.target: target,
                                               self.model.context: context})
            cost += np.sum(loss)

            predictions = sess.run(self.model.correct_prediction, feed_dict={self.model.aspect: aspect,
                                                                            self.model.loc: loc,
                                                                            self.model.target: target,
                                                                            self.model.context: context})
            for b in xrange(self.batch_size):
                if raw_labels[b] == predictions[b]:
                    acc = acc + 1

        return cost, acc / float(len(source_data))

    def set_variables(self, sess):
        emb_a = self.model.A.eval()
        emb_a[self.params['pad_idx'], :] = 0
        emb_b = self.model.B.eval()
        emb_b[self.params['pad_idx'], :] = 0
        emb_c = self.model.C.eval()
        emb_c[self.params['pad_idx'], :] = 0
        emb_ta = self.model.T_A.eval()
        emb_ta[self.params['pad_idx'], :] = 0
        emb_tb = self.model.T_B.eval()
        emb_tb[self.params['pad_idx'], :] = 0
        sess.run(self.model.A.assign(emb_a))
        sess.run(self.model.B.assign(emb_b))
        sess.run(self.model.C.assign(emb_c))
        sess.run(self.model.T_A.assign(emb_ta))
        sess.run(self.model.T_B.assign(emb_tb))