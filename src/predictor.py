"""

Neuron network of trend predictor.

Written by Guoqing Liu

"""

import numpy as np
import tensorflow as tf

class Predictor(object):
    """NN-based Trend Predictor"""
    def __init__(self, timesteps, num_channels, hidden_size, num_classes, lr_predictor, dropout):
        """
        """
        self.timesteps = timesteps
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lr = lr_predictor
        self.dropout = dropout
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
            self._predictor_nn()
            self._loss_train_op()
            self._reward_test_op()
            self._accuracy_test_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """ Input placeholders """

        self.keep_prob_ph = tf.placeholder(tf.float32)
        # bn_flag = tf.placeholder(tf.bool, name="training")
        self.input_ph = tf.placeholder(tf.float32, [None, self.timesteps, self.num_channels, 1],'input_x')
        # MLP
        # self.input_ph = tf.placeholder(tf.float32, [None, self.timesteps*self.num_channels],'input_x')
        self.label_ph = tf.placeholder(tf.float32, [None, self.num_classes], 'label_y')
        self.action_ph = tf.placeholder(tf.float32, [None, self.hidden_size], 'action_sequence')

    def _predictor_nn(self):
        """ Predictor network structure """
        self.conv1 = tf.layers.conv2d(inputs=self.input_ph, filters=512, kernel_size= [3,1], activation=tf.nn.relu)
        # self.conv = tf.layers.conv2d(inputs=self.input_ph, filters=16, kernel_size= [3,1], activation=tf.nn.sigmoid)
        # self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=16, kernel_size= [3,1], activation=tf.nn.relu)
        self.flat = tf.contrib.layers.flatten(self.conv1)
        self.dense0 = tf.layers.dense(inputs=self.flat, units=1024, activation=tf.nn.relu)
        self.dense1 = tf.layers.dense(inputs=self.dense0, units=512, activation=tf.nn.relu)
        self.dense = tf.layers.dense(inputs=self.dense1, units=self.hidden_size, activation=tf.nn.relu)
        # self.dense = tf.layers.dense(inputs=self.flat, units=self.hidden_size, activation=tf.nn.sigmoid)

        # MLP
        # self.dense = tf.layers.dense(inputs=self.input_ph, units=self.hidden_size, activation=tf.nn.relu)
        self.dense_dropout = tf.nn.dropout(self.dense, self.keep_prob_ph)
        self.dense_sel = self.dense_dropout*self.action_ph
        self.trend_prob = tf.layers.dense(inputs=self.dense_sel, units=self.num_classes, activation=tf.nn.softmax, name="decision_layer", use_bias=False)

    def _loss_train_op(self):
        self.predictor_vars = tf.trainable_variables()
        self.decision_vars = [var for var in self.predictor_vars if "decision_layer" in var.name]
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.label_ph*tf.log(self.trend_prob), reduction_indices=[1]))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.decision_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss, var_list= self.decision_vars)
        
    def _reward_test_op(self):
        self.reward = tf.reduce_sum(self.label_ph*tf.log(self.trend_prob), reduction_indices=[1])
        self.reward_acc = tf.cast(tf.equal(tf.argmax(self.trend_prob,1), tf.argmax(self.label_ph,1)), tf.float32)

    def _accuracy_test_op(self):
        self.acc_bool = tf.equal(tf.argmax(self.trend_prob,1), tf.argmax(self.label_ph,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.acc_bool, tf.float32))

    def _init_session(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.g, config=self.config)
        self.sess.run(self.init)

    def _get_hidden_state(self, batch_x):
        return self.sess.run(self.dense, feed_dict={self.input_ph: batch_x})

    # # placeholder
    # keep_prob = tf.placeholder(tf.float32)
    # # bn_flag = tf.placeholder(tf.bool, name="training")
    # x = tf.placeholder(tf.float32, [None, img_rows, img_cols, 1],'input_x')
    # y = tf.placeholder(tf.float32, [None, num_classes], 'label_y')

    # conv = tf.layers.conv2d(inputs=x, filters=16, kernel_size= [3,1], activation=None)
    # # conv_bn = tf.layers.batch_normalization(inputs=conv, training=bn_flag)
    # conv_out = tf.nn.relu(conv)
    # flat= tf.contrib.layers.flatten(conv_out)
    # dense = tf.layers.dense(inputs=flat, units=10, activation=None)
    # # dense_bn = tf.layers.batch_normalization(inputs=dense, training=bn_flag)
    # dense_output = tf.nn.relu(dense)
    # dense_dp = tf.nn.dropout(dense_output, keep_prob)
    # prediction = tf.layers.dense(inputs=dense_dp, units=3, activation=tf.nn.softmax)

    # # define loss
    # cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
    # # train_step = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cross_entropy_loss)
    # # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # # with tf.control_dependencies(update_ops):
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy_loss)

    # correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # train
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # sess = tf.InteractiveSession(config=config)

    # sess.run(tf.global_variables_initializer())
    # print("conv dimension:", sess.run([tf.shape(conv),tf.shape(flat)],feed_dict={x: X_test_reshape, y:Y_test}))