'''
Created on Oct/15/2018
@author: Bryce Xu
'''

import tensorflow as tf
import numpy as np
import time

from Data import DataGenerator

from BNN import Network


def training(X_train, Y_train, X_val, Y_val, lr_start, lr_end, epoch,
             batch_size):
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='xs')
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64, name='ys')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        learning_rate = tf.Variable(lr_start, name='learning_rate')
        # lr_decay = (lr_end / lr_start) ** (1 / epoch)
        lr_decay = 0.1
        lr_update = learning_rate.assign(learning_rate * lr_decay)

        train_data_generator = DataGenerator(X_train, Y_train)
        train_batch_generator = train_data_generator.dataGenerator(batch_size)
        iters = int(X_train.shape[0] / batch_size)
        print('number of batches for traning: {}'.format(iters))

        val_batch_size = 100
        val_data_generator = DataGenerator(X_val, Y_val)
        val_batch_generator = val_data_generator.dataGenerator(val_batch_size)
        print('data generator init')

        loss, error = Network(xs, ys,
                        is_training=is_training,
                        out_size=10)

        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        best_acc = 0
        cur_model_name = 'cifar-10_{}'.format(int(time.time()))

        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            sess.run(tf.global_variables_initializer(), {is_training: False})

            for epc in range(epoch):
                print("epoch {} ".format(epc + 1))
                # Training
                train_eve_sum = 0
                loss_sum = 0
                for _ in range(iters):
                    train_batch_x, train_batch_y = next(train_batch_generator)
                    _, cur_loss, train_err = sess.run([train, loss, error],
                                                      feed_dict={xs: train_batch_x, ys: train_batch_y,
                                                                 is_training: True})
                    train_eve_sum += np.sum(train_err)
                    loss_sum += np.sum(cur_loss)
                with tf.name_scope('train_accuracy'):
                    train_acc = 100 - train_eve_sum * 100 / Y_train.shape[0]

                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))

                # Validation
                valid_eve_sum = 0
                for _ in range(Y_val.shape[0] // val_batch_size):
                    val_batch_x, val_batch_y = next(val_batch_generator)
                    valid_eve, merge_result = sess.run([error, merge],
                                                       feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})
                    valid_eve_sum += np.sum(valid_eve)
                with tf.name_scope('validation_accuracy'):
                    valid_acc = 100 - valid_eve_sum * 100 / Y_val.shape[0]
                if epc == 20 or epc == 40 or epc == 60:
                    _lr = sess.run([lr_update])
                print('validation accuracy : {}%'.format(valid_acc))

                # When achieve the best validation accuracy, we store the model paramters
                if valid_acc > best_acc:
                    print('* Best accuracy: {}%'.format(valid_acc))
                    best_acc = valid_acc
                    # saver.save(sess, 'model/{}'.format(cur_model_name))
                print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))

            else:
                pass