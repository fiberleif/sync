"""

Neuron Network of Complex.

Written by Guoqing Liu

"""

import numpy as np
import tensorflow as tf

class Complex(object):
    """NN-based Complex"""
    def __init__(self, timesteps, num_features, hidden_size, num_classes):
        """
        """
        self.timesteps = timesteps
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.obs_his_dim = 2*hidden_size
        self._build_graph()
        self._init_session()

    # bulid graph
    def _build_graph(self):
        """
        Build and initialize Tensorflow graph

        """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._complex_nn()
            self._loss_train_op()
            self._reward_test_op()
            self._accuracy_test_op()
            self._sample_multinomial()
            self._sample_maximum()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """ Input placeholders """

        # Encoder placeholder
        self.input_ph = tf.placeholder(tf.float32, [None, self.timesteps, self.num_features, 1],'input_x')
        self.lr_en_ph = tf.placeholder(tf.float32, (), 'learning_rate4encoder')

        # MLP optional
        # self.input_ph = tf.placeholder(tf.float32, [None, self.timesteps*self.num_features],'input_x')

        # Predictor placeholder
        self.keep_prob_ph = tf.placeholder(tf.float32, (), 'keep_prob4dropout')
        self.action_ph = tf.placeholder(tf.float32, [None, self.hidden_size], 'action_sequence')
        self.label_ph = tf.placeholder(tf.float32, [None, self.num_classes], 'label_y')
        self.lr_pre_ph = tf.placeholder(tf.float32, (), 'learning_rate4predictor')

        # Selector placeholder
        self.obs_his_ph = tf.placeholder(tf.float32, [None, self.obs_his_dim], "observation_addition") 
        self.act_ph = tf.placeholder(tf.float32, [None, 2], "action") 
        self.adv_ph = tf.placeholder(tf.float32, [None,], "advantage")
        self.lr_sel_ph = tf.placeholder(tf.float32, (), 'learning_rate4selector')

    def _complex_nn(self):
        """ Build network structure """

        # Encoder nn
        with tf.variable_scope('Encoder'):
            self.conv1 = tf.layers.conv2d(inputs=self.input_ph, filters=16, kernel_size= [3,1], activation=tf.nn.relu)
            self.flat = tf.contrib.layers.flatten(self.conv1)
            self.dense = tf.layers.dense(inputs=self.flat, units=self.hidden_size, activation=tf.nn.relu)

        # MLP optional
        # self.dense = tf.layers.dense(inputs=self.input_ph, units=self.hidden_size, activation=tf.nn.relu)

        # Predictor nn
        with tf.variable_scope('Predictor'):
            self.dense_dropout = tf.nn.dropout(self.dense, self.keep_prob_ph)
            self.dense_sel = self.dense_dropout*self.action_ph
            self.trend_prob = tf.layers.dense(inputs=self.dense_sel, units=self.num_classes, activation=tf.nn.softmax, name="decision_layer", use_bias=False)

        # Selector nn
        with tf.variable_scope('Selector'):
            self.obs = tf.concat([self.dense, self.obs_his_ph], 1)
            self.hidden = tf.layers.dense(inputs=self.obs, units=128, activation=tf.nn.relu)
            self.act_prob = tf.layers.dense(inputs=self.dense, units=2, activation=tf.nn.softmax)

    def _loss_train_op(self):
        """ Build loss functions """

        # First term in loss4encoder 
        self.loss_en1 = tf.reduce_sum(-tf.reduce_sum(self.label_ph*tf.log(self.trend_prob), reduction_indices=[1]))
        # Second term in loss4encoder
        self.loss_en2 = tf.reduce_sum(-tf.reduce_sum(self.act_ph*tf.log(self.act_prob), reduction_indices=[1])*self.adv_ph)
        self.loss_pre = tf.reduce_mean(-tf.reduce_sum(self.label_ph*tf.log(self.trend_prob), reduction_indices=[1]))
        self.loss_sel = tf.reduce_mean(-tf.reduce_sum(self.act_ph*tf.log(self.act_prob), reduction_indices=[1])*self.adv_ph)

        # Build Optimizers
        optimizer_en_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr_en_ph)
        optimizer_pre_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pre_ph)
        optimizer_sel_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr_sel_ph)

        # Training Variables for each Optimizer
        en_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
        pre_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
        sel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Selector')

        # Create gradient Operations
        self.gradient_en1 = optimizer_en_op.compute_gradients(self.loss_en1, var_list=en_vars)
        self.gradient_en2 = optimizer_en_op.compute_gradients(self.loss_en2, var_list=en_vars)

        grads_ph = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
                     for (g, v) in self.gradient_en1]

        self.train_en_op = optimizer_en_op.apply_gradients(grads_ph)
        self.train_pre_op = optimizer_pre_op.minimize(self.loss_pre, var_list=pre_vars)
        self.train_sel_op = optimizer_sel_op.minimize(self.loss_sel, var_list=sel_vars)
                
    def _reward_test_op(self):
        """ Reward computation for Selector """
        self.reward = tf.reduce_sum(self.label_ph*tf.log(self.trend_prob), reduction_indices=[1])
        self.reward_acc = tf.cast(tf.equal(tf.argmax(self.trend_prob,1), tf.argmax(self.label_ph,1)), tf.float32)

    def _accuracy_test_op(self):
        """ Accuracy computation for Predictor """
        self.acc_bool = tf.equal(tf.argmax(self.trend_prob,1), tf.argmax(self.label_ph,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.acc_bool, tf.float32))

    def _init_session(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.g, config=self.config)
        self.sess.run(self.init)

    def _get_hidden_state(self, batch_x):
        return self.sess.run(self.dense, feed_dict={self.input_ph: batch_x})

    def _sample_multinomial(self):
        """ Sample from distribution, given observation """
        """ Return col vector: [batch_size, num_samples]"""
        self.sample_mul = tf.multinomial(self.act_prob, 1)
        
    def _sample_maximum(self):
        """ Sample via argmax, given observation """
        """ Return row vector"""
        self.sample_max = tf.argmax(self.act_prob, 1)

    def sample_multinomial(self, batch_x, observation):
        return self.sess.run(self.sample_mul, feed_dict={self.input_ph: batch_x, self.obs_his_ph: observation})

    def sample_maximum(self, batch_x, observation):
        return self.sess.run(self.sample_max, feed_dict={self.input_ph: batch_x, self.obs_his_ph: observation})
        