"""
multi timestep stock prediction via Sparse Feature Extraction.

Tensorflow implementation

Written by Guoqing Liu 

"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
import tensorflow as tf
from functools import reduce
from operator import mul

def main():
    
    np.random.seed(0)
    num_classes = 3 
    batch_size = 256
    origin_length = 39

    # load dataset
    timestep = int(sys.argv[1])
    epochs = int(sys.argv[2])
    lr = float(sys.argv[3])
    dropout = float(sys.argv[4])

    feature_start_column_idx = 2
    feature_end_column_idx = 2 + origin_length*timestep
    label_column_idx = feature_end_column_idx + 1

    train_dataset_path = "/home/data/guoqing/dataset/timestep_" + str(timestep) \
                + "_train_dataset.csv"
    test_dataset_path = "/home/data/guoqing/dataset/timestep_" + str(timestep) \
                + "_test_dataset.csv"
    save_path = "/home/data/guoqing/prediction/result/cnn_timestep_" + str(timestep) \
                 + "_lr_" + str(lr) + "_dp_" + str(dropout) + "_epoch_" + str(epochs) + ".csv"


    train_csv = pd.read_csv(train_dataset_path)
    test_csv = pd.read_csv(test_dataset_path)
    train_dataset = train_csv.values
    test_dataset = test_csv.values

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    X_train = train_dataset[:,feature_start_column_idx:feature_end_column_idx].astype(np.float)
    Y_train = train_dataset[:,label_column_idx]
    train_size = Y_train.shape[0]
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    X_test = test_dataset[:,feature_start_column_idx:feature_end_column_idx].astype(np.float)
    Y_test = test_dataset[:, label_column_idx]
    test_size = Y_test.shape[0]
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    # normalization
    factor_scaler = preprocessing.MinMaxScaler()
    factor_scaler.fit(X_train)
    X_train_norm = factor_scaler.transform(X_train)
    X_test_norm = factor_scaler.transform(X_test)

    # reshape
    img_rows = timestep
    img_cols = origin_length
    X_train_reshape = X_train_norm.reshape(X_train_norm.shape[0], img_rows, img_cols, 1)
    X_test_reshape = X_test_norm.reshape(X_test_norm.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # tensorflow
    # bulid graph

    # placeholder
    keep_prob = tf.placeholder(tf.float32)
    # bn_flag = tf.placeholder(tf.bool, name="training")
    x = tf.placeholder(tf.float32, [None, img_rows, img_cols, 1],'input_x')
    y = tf.placeholder(tf.float32, [None, 3], 'label_y')

    conv = tf.layers.conv2d(inputs=x, filters=16, kernel_size= [3,1], activation=None)
    # conv_bn = tf.layers.batch_normalization(inputs=conv, training=bn_flag)
    conv_out = tf.nn.relu(conv)
    flat= tf.contrib.layers.flatten(conv_out)
    dense = tf.layers.dense(inputs=flat, units=10, activation=None)
    # dense_bn = tf.layers.batch_normalization(inputs=dense, training=bn_flag)
    dense_output = tf.nn.relu(dense)
    dense_dp = tf.nn.dropout(dense_output, keep_prob)
    prediction = tf.layers.dense(inputs=dense_dp, units=3, activation=tf.nn.softmax)

    # define loss
    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
    # train_step = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cross_entropy_loss)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy_loss)

    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)

    sess.run(tf.global_variables_initializer())
    # print("conv dimension:", sess.run([tf.shape(conv),tf.shape(flat)],feed_dict={x: X_test_reshape, y:Y_test}))

    num_params = 0
    for variable in tf.trainable_variables():
        print(variable.name)
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    print("num_params", num_params)

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    train_batch_num = train_size // batch_size
    test_batch_num = test_size // batch_size
    # print(train_size)

    with sess.as_default():
        for e in range(epochs):
            # shuffle training data
            shuffle_indices = np.random.permutation(np.arange(train_size))
            X_train_shuffle = X_train_reshape[shuffle_indices]
            Y_train_shuffle = Y_train[shuffle_indices]
            train_acc_sum = 0
            test_acc_sum = 0 
            train_loss_sum = 0
            test_loss_sum = 0

            for i in range(0, train_batch_num):
                start = i * batch_size
                # batch_x = X_train_shuffle[start:start + batch_size]
                # batch_y = Y_train_shuffle[start:start + batch_size]
                batch_x = X_train_reshape[start:start + batch_size]
                batch_y = Y_train[start:start + batch_size]
                train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob:1})
                train_loss = cross_entropy_loss.eval(feed_dict={x: batch_x, y: batch_y, keep_prob:1})
                train_acc_sum += train_acc
                train_loss_sum += train_loss
            
            train_acc_avg = train_acc_sum / train_batch_num
            train_loss_avg = train_loss_sum / train_batch_num 
            train_acc_list.append(train_acc_avg)
            train_loss_list.append(train_loss_avg)

            for i in range(0, test_batch_num):
                start = i * batch_size
                # batch_x = X_train_shuffle[start:start + batch_size]
                # batch_y = Y_train_shuffle[start:start + batch_size]
                batch_x = X_test_reshape[start:start + batch_size]
                batch_y = Y_test[start:start + batch_size]
                test_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob:1})
                test_loss = cross_entropy_loss.eval(feed_dict={x: batch_x, y: batch_y, keep_prob:1})
                test_acc_sum += test_acc
                test_loss_sum += test_loss
            
            test_acc_avg = test_acc_sum / test_batch_num
            test_loss_avg = test_loss_sum / test_batch_num 
            test_acc_list.append(test_acc_avg)
            test_loss_list.append(test_loss_avg)

            print("epoch %d: train_acc %f, test_acc %f, train_loss %f, test_loss %f" % (e, train_acc_avg, test_acc_avg, train_loss_avg, test_loss_avg))

            #Minibatch training
            for i in range(0, train_size // batch_size):
                start = i * batch_size
                # batch_x = X_train_shuffle[start:start + batch_size]
                # batch_y = Y_train_shuffle[start:start + batch_size]
                batch_x = X_train_shuffle[start:start + batch_size]
                batch_y = Y_train_shuffle[start:start + batch_size]
                # run optimizer with batch
                # sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

    result = pd.DataFrame({'train_acc': train_acc_list, 'test_acc': test_acc_list, 'train_loss': train_loss_list ,'test_loss': test_loss_list})
    result.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
    sys.exit()
