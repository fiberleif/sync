"""

Neuron network of pattern selector.

Written by Guoqing Liu

"""

import numpy as np
import tensorflow as tf

class Selector(object):
    """NN-based Pattern Selector """
    def __init__(self, hidden_size, lr_selector):
        self.input_dim = 3*hidden_size
        self.lr = lr_selector
        self._build_graph()
        self._init_session()
    
    def _build_graph(self):
        """ Construct tensorflow graph, including loss function, init op and train op """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._selector_nn()
            self._loss_train_op()
            self._sample_multinomial()
            self._sample_maximum()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """" Input placeholders """
        # inputs, labels and actions:
        self.lr_ph = tf.placeholder(tf.float32, (), 'learning_rate')
        self.obs_ph = tf.placeholder(tf.float32, [None, self.input_dim], "observations") 
        self.act_ph = tf.placeholder(tf.float32, [None, 2], "actions") 
        self.adv_ph = tf.placeholder(tf.float32, [None,], "advantages")

    def _selector_nn(self):
        self.dense = tf.layers.dense(inputs=self.obs_ph, units=128, activation=tf.nn.relu)
        self.act_prob = tf.layers.dense(inputs=self.dense, units=2, activation=tf.nn.softmax)

    def _loss_train_op(self):
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.act_ph*tf.log(self.act_prob), reduction_indices=[1])*self.adv_ph)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr_ph).minimize(self.loss)
    
    def _sample_multinomial(self):
        """ Sample from distribution, given observation """
        """ Return col vector: [batch_size, num_samples]"""
        self.sample_mul = tf.multinomial(self.act_prob, 1)
        
    def _sample_maximum(self):
        """ Sample via argmax, given observation """
        """ Return row vector"""
        self.sample_max = tf.argmax(self.act_prob, 1)
    
    def _init_session(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.g, config=self.config)
        self.sess.run(self.init)

    def sample_multinomial(self, observation):
        return self.sess.run(self.sample_mul, feed_dict={self.obs_ph: observation})

    def sample_maximum(self, observation):
        return self.sess.run(self.sample_max, feed_dict={self.obs_ph: observation})

