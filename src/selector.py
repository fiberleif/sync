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
            # self._sample()
            self._sample_multinomial()
            self._sample_maximum()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """" Input placeholders """
        # inputs, labels and actions:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.input_dim), "observations") 
        self.act_ph = tf.placeholder(tf.float32, (None, 2), "actions") 
        self.adv_ph = tf.placeholder(tf.float32, (None,), "advantages")

    def _selector_nn(self):
        self.act_prob = tf.layers.dense(inputs=self.obs_ph, units=2, activation=tf.nn.softmax)

    def _loss_train_op(self):
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.act_ph*tf.log(self.act_prob), reduction_indices=[1])*self.adv_ph)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    def _sample_multinomial(self):
        """ Sample from distribution, given observation """
        self.sample_mul = tf.multinomial(self.act_prob, 1)
        
    def _sample_maximum(self):
        """ Sample via argmax, given observation """
        """ return other type with comparison to multinomial"""
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




 # # placeholder
 #    keep_prob = tf.placeholder(tf.float32)
 #    # bn_flag = tf.placeholder(tf.bool, name="training")
 #    x = tf.placeholder(tf.float32, [None, img_rows, img_cols, 1],'input_x')
 #    y = tf.placeholder(tf.float32, [None, num_classes], 'label_y')

 #    conv = tf.layers.conv2d(inputs=x, filters=16, kernel_size= [3,1], activation=None)
 #    # conv_bn = tf.layers.batch_normalization(inputs=conv, training=bn_flag)
 #    conv_out = tf.nn.relu(conv)
 #    flat= tf.contrib.layers.flatten(conv_out)
 #    dense = tf.layers.dense(inputs=flat, units=10, activation=None)
 #    # dense_bn = tf.layers.batch_normalization(inputs=dense, training=bn_flag)
 #    dense_output = tf.nn.relu(dense)
 #    dense_dp = tf.nn.dropout(dense_output, keep_prob)
 #    prediction = tf.layers.dense(inputs=dense_dp, units=3, activation=tf.nn.softmax)

 #    # define loss
 #    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
 #    # train_step = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cross_entropy_loss)
 #    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
 #    # with tf.control_dependencies(update_ops):
 #    train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy_loss)

 #    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
 #    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 #    # train
 #    config = tf.ConfigProto()
 #    config.gpu_options.allow_growth=True
 #    sess = tf.InteractiveSession(config=config)

 #    sess.run(tf.global_variables_initializer())
 #    print("conv dimension:", sess.run([tf.shape(conv),tf.shape(flat)],feed_dict={x: X_test_reshape, y:Y_test}))