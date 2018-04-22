"""
multi timestep stock prediction via Sparse Feature Extraction.

Tensorflow implementation

Written by Guoqing Liu 

"""
import os
import sys
import keras
import signal
import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from predictor import Predictor
from selector import Selector
from datetime import datetime
from functools import reduce
from operator import mul
from sklearn import preprocessing

def prepare_data(timesteps, num_features, num_classes, train_dataset_path, test_dataset_path):
    """
    Prepare training data.

    Args:
    pass

    """
    feature_start_column_idx = 2
    feature_end_column_idx = 2 + num_features * timesteps
    label_column_idx = feature_end_column_idx + 1

    train_csv = pd.read_csv(train_dataset_path)
    test_csv = pd.read_csv(test_dataset_path)
    train_dataset = train_csv.values
    test_dataset = test_csv.values

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
    X_train_reshape = X_train_norm.reshape(X_train_norm.shape[0], timesteps, num_features, 1)
    X_test_reshape = X_test_norm.reshape(X_test_norm.shape[0], timesteps, num_features, 1)

    # # MLP
    # X_train_reshape = X_train_norm.reshape(X_train_norm.shape[0], timesteps*num_features)
    # X_test_reshape = X_test_norm.reshape(X_test_norm.shape[0], timesteps*num_features)
    # input_shape = (img_rows, img_cols, 1)
    return X_train_reshape, Y_train, X_test_reshape, Y_test, train_size, test_size


def run_episode(batch_x, predictor, selector, batch_size, hidden_size):
    """
    collect trajectories given batch_x
    """
    hidden_vector = predictor._get_hidden_state(batch_x)
    binary_vector = np.zeros((batch_size, hidden_size))
    state_list = []
    action_list = []
    for i in range(hidden_size):
        position_vector = np.zeros((batch_size, hidden_size))
        position_vector[:,i] = 1
        state = np.concatenate((hidden_vector, binary_vector, position_vector), axis=1)
        action = selector.sample_multinomial(state).reshape((batch_size, 1))
        state_list.append(state)
        action_list.append(action)
        binary_vector[:,i] = action[:,0]
    # observations = np.concatenate(state_list, axis=1)
    actions = np.concatenate(action_list, axis=1)
    observations_ps = np.concatenate(state_list, axis=0)
    actions_pre = np.concatenate(action_list, axis=0)

    # convert actions from (batch_size*hidden_size, 1) to (batch_size*hidden_size, 2)
    actions_ps = np.zeros((batch_size*hidden_size,2))
    for i in range(actions_pre.shape[0]):
        if(actions_pre[i,0] == 1):
            actions_ps[i, 1] = 1
        else:
            actions_ps[i, 0] = 1

    return actions, observations_ps, actions_ps


def main(timesteps, num_features, hidden_size, num_classes, batch_size, epochs, lr_predictor, lr_selector, dropout):
    """ Main training loop.

    Args:
    pass

    """

    """
    Set global seed function
    pass

    """
    train_dataset_path = "/home/data/guoqing/dataset/timestep_" + str(timesteps) \
                + "_train_dataset.csv"
    test_dataset_path = "/home/data/guoqing/dataset/timestep_" + str(timesteps) \
                + "_test_dataset.csv"
    save_path = "/home/data/guoqing/prediction/result/model_1cnn_1dense_timestep_" + str(timesteps) \
                     + "_selector_lr_" + str(lr_selector) + "_reward_logp_bias.csv"

    update_len = 20
    predictor_len = 9
    # Prepare train data.
    X_train_reshape, Y_train, X_test_reshape, Y_test, train_size, test_size = prepare_data(timesteps, num_features, \
                                                    num_classes, train_dataset_path, test_dataset_path)

    # Build networks structure, including variables, operation.
    predictor = Predictor(timesteps, num_features, hidden_size, num_classes, lr_predictor, dropout)
    selector = Selector(hidden_size, lr_selector)

    # Global variable statistic
    with predictor.g.as_default():
        num_params = 0
        for variable in tf.trainable_variables():
            print(variable.name)
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print("predictor num_params", num_params)

    with selector.g.as_default():
        num_params = 0
        for variable in tf.trainable_variables():
            print(variable.name)
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print("selector num_params", num_params)

    # Pre-train trend predictor
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    train_spa_list = []
    test_spa_list = []
    train_reward_list = []
    test_acc_baseline_list = []

    train_batch_num = train_size // batch_size
    test_batch_num = test_size // batch_size
    print("train batch number:", train_batch_num)
    print("test batch number:", test_batch_num)
   
    for e in range(20):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(train_size))
        X_train_shuffle = X_train_reshape[shuffle_indices]
        Y_train_shuffle = Y_train[shuffle_indices]
        train_acc_sum = 0
        test_acc_sum = 0 
        train_loss_sum = 0
        test_loss_sum = 0
        
        # Testing for train dataset
        for i in range(0, train_batch_num):
            start = i * batch_size
            # batch_x = X_train_shuffle[start:start + batch_size]
            # batch_y = Y_train_shuffle[start:start + batch_size]
            batch_x = X_train_reshape[start:start + batch_size]
            batch_y = Y_train[start:start + batch_size]
            batch_a = np.ones((batch_size, hidden_size))

            # train_acc = predictor.sess.run(predictor.accuracy, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})
            # train_loss = predictor.sess.run(predictor.loss, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            train_acc, train_loss = predictor.sess.run([predictor.accuracy, predictor.loss], feed_dict={predictor.input_ph: batch_x, \
                                                    predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            train_acc_sum += train_acc
            train_loss_sum += train_loss
        
        train_acc_avg = train_acc_sum / train_batch_num
        train_loss_avg = train_loss_sum / train_batch_num 
        train_acc_list.append(train_acc_avg)
        train_loss_list.append(train_loss_avg)

        # Testing for test dataset
        for i in range(0, test_batch_num):
            start = i * batch_size
            # batch_x = X_train_shuffle[start:start + batch_size]
            # batch_y = Y_train_shuffle[start:start + batch_size]
            batch_x = X_test_reshape[start:start + batch_size]
            batch_y = Y_test[start:start + batch_size]
            batch_a = np.ones((batch_size, hidden_size))

            # test_acc = predictor.sess.run(predictor.accuracy, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})
            # test_loss = predictor.sess.run(predictor.loss, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            test_acc, test_loss = predictor.sess.run([predictor.accuracy, predictor.loss], feed_dict={predictor.input_ph: batch_x, \
                                                    predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            test_acc_sum += test_acc
            test_loss_sum += test_loss
        
        test_acc_avg = test_acc_sum / test_batch_num
        test_loss_avg = test_loss_sum / test_batch_num 
        test_acc_list.append(test_acc_avg)
        test_loss_list.append(test_loss_avg)
        train_spa_list.append(1)
        test_spa_list.append(1)
        train_reward_list.append(0)
        test_acc_baseline_list.append(test_acc_avg)

        print("epoch %d: train_acc %f, test_acc %f, train_loss %f, test_loss %f" % (e, train_acc_avg, test_acc_avg, train_loss_avg, test_loss_avg))


        # Minibatch training
        for i in range(0, train_batch_num):
            start = i * batch_size
            batch_x = X_train_shuffle[start:start + batch_size]
            batch_y = Y_train_shuffle[start:start + batch_size]
            batch_a = np.ones((batch_size, hidden_size))

            predictor.sess.run(predictor.train_op, feed_dict={predictor.input_ph: batch_x, \
                                            predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: dropout})


    # Pre-train pattern selector
    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(train_size))
        X_train_shuffle = X_train_reshape[shuffle_indices]
        Y_train_shuffle = Y_train[shuffle_indices]
        train_acc_sum = 0
        test_acc_sum = 0
        test_acc_baseline_sum = 0 
        train_loss_sum = 0
        test_loss_sum = 0
        train_spa_sum = 0
        test_spa_sum = 0
        train_reward_sum = 0

        # Testing for train dataset
        for i in range(0, train_batch_num):
            start = i * batch_size
            batch_x = X_train_reshape[start:start + batch_size]
            batch_y = Y_train[start:start + batch_size]
            batch_a, _, _ = run_episode(batch_x, predictor, selector, batch_size, hidden_size)

            # train_acc = predictor.sess.run(predictor.accuracy, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})
            # train_loss = predictor.sess.run(predictor.loss, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            train_acc, train_loss = predictor.sess.run([predictor.accuracy, predictor.loss], feed_dict={predictor.input_ph: batch_x, \
                                                    predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            train_acc_sum += train_acc
            train_loss_sum += train_loss
            train_spa_sum += np.mean(batch_a)
        
        train_acc_avg = train_acc_sum / train_batch_num
        train_loss_avg = train_loss_sum / train_batch_num
        train_spa_avg = train_spa_sum / train_batch_num
        train_acc_list.append(train_acc_avg)
        train_loss_list.append(train_loss_avg)
        train_spa_list.append(train_spa_avg)

        # Testing for test dataset
        for i in range(0, test_batch_num):
            start = i * batch_size
            batch_x = X_test_reshape[start:start + batch_size]
            batch_y = Y_test[start:start + batch_size]

            batch_a, _, _ = run_episode(batch_x, predictor, selector, batch_size, hidden_size)
            
            batch_one = np.ones((batch_size, hidden_size))

            test_acc_baseline = predictor.sess.run(predictor.accuracy, feed_dict={predictor.input_ph: batch_x, \
                                                    predictor.label_ph: batch_y, predictor.action_ph: batch_one, predictor.keep_prob_ph: 1})

            # test_acc = predictor.sess.run(predictor.accuracy, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})
            # test_loss = predictor.sess.run(predictor.loss, feed_dict={predictor.input_ph: batch_x, \
            #                                         predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            test_acc, test_loss = predictor.sess.run([predictor.accuracy, predictor.loss], feed_dict={predictor.input_ph: batch_x, \
                                                    predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})

            test_acc_sum += test_acc
            test_loss_sum += test_loss
            test_acc_baseline_sum += test_acc_baseline
            test_spa_sum += np.mean(batch_a)
        
        test_acc_avg = test_acc_sum / test_batch_num
        test_loss_avg = test_loss_sum / test_batch_num
        test_spa_avg = test_spa_sum / test_batch_num
        test_acc_baseline_avg = test_acc_baseline_sum / test_batch_num
        test_acc_list.append(test_acc_avg)
        test_loss_list.append(test_loss_avg)
        test_spa_list.append(test_spa_avg)
        test_acc_baseline_list.append(test_acc_baseline_avg)

        print("epoch %d: train_acc %f, test_acc %f, test_baseline_acc %f, train_loss %f, test_loss %f, train_spa %f, test_spa %f" % \
                                                (e, train_acc_avg, test_acc_avg, test_acc_baseline_avg, train_loss_avg, test_loss_avg, train_spa_avg, test_spa_avg))

        # Minibatch training
        for i in range(0, train_batch_num):
            start = i * batch_size
            batch_x = X_train_shuffle[start:start + batch_size]
            batch_y = Y_train_shuffle[start:start + batch_size]

            # record states, actions.
            batch_a, observations_ps, actions_ps = run_episode(batch_x, predictor, selector, batch_size, hidden_size)

            batch_reward = predictor.sess.run(predictor.reward, feed_dict={predictor.input_ph: batch_x, \
                                            predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})
            
            # # 1/0->0.9/0.1
            # trans_count = 0
            # for index in range(batch_size):
            #     if(batch_reward[index] == 1):
            #         batch_reward[index] = 0.9
            #         trans_count += 1
            #     elif(batch_reward[index] == 0):
            #         batch_reward[index] = 0.1
            #         trans_count += 1
            # assert trans_count== batch_size

            # batch_reward = batch_reward.reshape((batch_size, -1))

            # # advice from teacher xia
            # batch_prob = predictor.sess.run(predictor.trend_prob, feed_dict={predictor.input_ph: batch_x, \
            #                                 predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1})
            # reward_basic = np.empty([batch_size, num_classes])
            # for row in range(batch_size):
            #     for col in range(num_classes):
            #         if(batch_y[row, col]==1):
            #             reward_basic[row, col] = 0.9
            #         elif(batch_y[row, col] == 0):
            #             reward_basic[row, col] = 0.1
            # batch_reward = np.sum(reward_basic*batch_prob, axis=1)


            # Reward shaping here
            # v1-all baseline
            batch_one = np.ones((batch_size, hidden_size))
            batch_reward_bias = predictor.sess.run(predictor.reward, feed_dict={predictor.input_ph: batch_x, \
                                            predictor.label_ph: batch_y, predictor.action_ph: batch_one, predictor.keep_prob_ph: 1})
            batch_reward -= batch_reward_bias
            
            """
            # v2-batch baseline
            batch_reward_bias = np.mean(batch_reward)
            batch_reward -= batch_reward_bias
            """

            train_reward = np.mean(batch_reward)
            train_reward_sum += train_reward

            extended_reward = np.tile(batch_reward, hidden_size)
            
            # if(e % update_len <= predictor_len):
            #     predictor.sess.run(predictor.train_op, feed_dict={predictor.input_ph: batch_x, \
            #                                 predictor.label_ph: batch_y, predictor.action_ph: batch_a, predictor.keep_prob_ph: 1}) 
            # else:
            #     selector.sess.run(selector.train_op, feed_dict={selector.obs_ph: observations_ps, \
            #                                 selector.act_ph: actions_ps, selector.adv_ph: extended_reward}) 
            
            # Warm-start for selector
            selector.sess.run(selector.train_op, feed_dict={selector.obs_ph: observations_ps, \
                                            selector.act_ph: actions_ps, selector.adv_ph: extended_reward}) 
        train_reward_avg = train_reward_sum / train_batch_num
        train_reward_list.append(train_reward_avg)
        print("epoch %d: train_reward %f" % (e, train_reward_avg))

    result = pd.DataFrame({'train_acc': train_acc_list, 'test_acc': test_acc_list, 'test_baseline_acc': test_acc_baseline_list, 'train_loss': train_loss_list ,'test_loss': test_loss_list, \
                                                            'train_spa': train_spa_list, 'test_spa': test_spa_list, 'train_reward': train_reward_list})
    result.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, help='input sequence timesteps', default=5)
    parser.add_argument('--num_features', type=int, help='input feature dimension', default=39)
    parser.add_argument('--hidden_size', type=int, help='hidden size for selection', default=10)
    parser.add_argument('--num_classes', type=int, help='number of class', default=3)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--epochs', type=int, help='epoch number', default=100)
    parser.add_argument('--lr_predictor', type=float, help='learning rate of predictor network', default=1e-1)
    parser.add_argument('--lr_selector', type=float, help='learning rate of selector network', default=1e-2)
    parser.add_argument('--dropout', type=float, help='keep rate', default=0.5)

    args = parser.parse_args()
    main(**vars(args))
    sys.exit()
