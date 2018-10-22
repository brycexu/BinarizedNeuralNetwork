'''
Created on Oct/15/2018
@author: Bryce Xu
'''

import tensorflow as tf
import numpy as np

'''
ResNet-18
filters=[64,64,128,256,512]
kernels=[3,3,3,3,3]
strides=[1,1,2,2,2]
Input:
    Cifar-10:32x32x3
Con1:
    Conv:3,64,k=3,s=1,p=1
    BatchNorm
    ReLu
Layer2:
    Residual:
        shortcut=x
        Conv:3,64,k=3,s=1,p=1
        BatchNorm
        ReLu
        Conv:3,64,k=3,s=1,p=1
        BatchNorm
        +shortcut
        ReLu
    Residual:
        repeat
Layer3:
    Residual_first
    Residual
Layer4:
    Residual_first
    Residual
Layer5:
    Residual_first
    Residual
FC6:
    MAXPOOL(4)
    FC    

'''

class First_Layer(object):
    def __init__(self, input_X, out_size, is_training=None, index = 0):
        with tf.name_scope('FL_%d' % index):
            with tf.variable_scope('FL_%d' % index):

                in_size = input_X.shape[3]

                x_1 = CNN(input_X, in_size, out_size, 1, 'SAME', index).output()
                x_2 = BN(x_1, is_training, index).output()
                cell_out = tf.nn.relu(x_2)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out


class Residual_Block_First(object):
    def __init__(self, input_X, out_size, strides, is_training=None, index = 0):
        with tf.name_scope('RB_F_%d' % index):
            with tf.variable_scope('RB_F_%d' % index):

                in_size = input_X.shape[3]

                # Shortcut connection
                if in_size == out_size:
                    if strides == 1:
                        shortcut = tf.identity(input_X)
                else:
                    shortcut = CNN(input_X, in_size, out_size, strides, 'SAME', index+22).output()

                # Residual
                x_1 = CNN(input_X, in_size, out_size, strides, 'SAME', index).output()
                x_2 = BN(x_1, is_training, index).output()
                x_3 = tf.nn.relu(x_2)
                x_4 = CNN(x_3, out_size, out_size, 1, 'SAME', index+11).output()
                x_5 = BN(x_4, is_training, index+11).output()

                # Merge
                x_6 = x_5 + shortcut
                cell_out = tf.nn.relu(x_6)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out


class Residual_Block(object):
    def __init__(self, input_X, out_size, strides, is_training=None, index = 0):
        with tf.name_scope('RB_%d' % index):
            with tf.variable_scope('RB_%d' % index):

                in_size = input_X.shape[3]

                # Shortcut connection
                shortcut = input_X

                # Residual
                x_1 = CNN(input_X, in_size, out_size, strides, 'SAME', index).output()
                x_2 = BN(x_1, is_training, index).output()
                x_3 = tf.nn.relu(x_2)
                x_4 = CNN(x_3, out_size, out_size, 1, 'SAME', index+11).output()
                x_5 = BN(x_4, is_training, index+11).output()

                # Merge
                x_6 = x_5 + shortcut
                cell_out = tf.nn.relu(x_6)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out

'''
class BN(object):
    def __init__(self, input_X, is_training, index = 0):
        moving_average_dacay = 0.99
        with tf.name_scope('BN_%d' % index):
            with tf.variable_scope('BN_%d' % index):
                decay = moving_average_dacay
                batch_mean, batch_var = tf.nn.moments(input_X,[0,1,2])
                with tf.device('/CPU:0'):
                    mu = tf.get_variable('mu_%d' % index, batch_mean.get_shape(), tf.float32,
                                         initializer=tf.zeros_initializer(), trainable=False)
                    sigma = tf.get_variable('sigma_%d' % index, batch_var.get_shape(), tf.float32,
                                            initializer=tf.ones_initializer(), trainable=False)
                    beta = tf.get_variable('beta_%d' % index, batch_mean.get_shape(), tf.float32,
                                           initializer=tf.zeros_initializer())
                    gamma = tf.get_variable('gamma_%d' % index, batch_var.get_shape(), tf.float32,
                                            initializer=tf.ones_initializer())
                update = 1.0 - decay
                update_mu = mu.assign_sub(update * (mu - batch_mean))
                update_sigma = sigma.assign_sub(update * (sigma - batch_var))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)
                mean, var = tf.cond(is_training, lambda: (batch_mean, batch_var),
                                    lambda: (mu, sigma))
                cell_out = tf.nn.batch_normalization(input_X, mean, var, beta, gamma, 1e-4)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out

'''


class BN(object):
    def __init__(self, input_X, is_training, index = 0):
        with tf.name_scope('BN_%d' % index):
            with tf.variable_scope('BN_Weights_%d' % index):
                batch_mean, batch_var = tf.nn.moments(input_X, [0,1,2], keep_dims = True)
                shift = tf.Variable(tf.zeros([input_X.shape[3]]))
                scale = tf.Variable(tf.ones([input_X.shape[3]]))
                epsilon = 1e-4
                cell_out = tf.nn.batch_normalization(input_X, batch_mean, batch_var, shift, scale, epsilon)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out


class CNN(object):
    def __init__(self, input_X, in_size, out_size, strides, padding, index = 0):
        with tf.name_scope('CNN_%d' % index):
            with tf.variable_scope('CNN_Weights_%d' % index):
                W_shape = [3,3,in_size,out_size]
                Weight = tf.get_variable(name='CNN_Layer_Weights_%d' % index, shape=W_shape,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d())
                self.Weight = Weight
                cell_out = tf.nn.conv2d(input_X, Weight, strides = [1,strides,strides,1], padding = padding)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out


class FC(object):
    def __init__(self, input_X, in_size, out_size, index = 0):
        with tf.name_scope('FC_%d' % index):
            with tf.variable_scope('FC_Weights_%d' % index):
                W_shape = [in_size, out_size]
                Weight = tf.get_variable(name='FC_Layer_Weights_%d' % index, shape=W_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
                self.Weight = Weight
                bias_shape = [out_size]
                bias = tf.get_variable(name='FC_Layer_Bias_%d' % index, shape=bias_shape,
                                       initializer=tf.contrib.layers.xavier_initializer())
                cell_out = tf.add(tf.matmul(input_X, self.Weight), bias)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out


class MaxPool_Layer(object):
    def __init__(self, input_X, index = 0):
        with tf.name_scope('MP_CNN_%d' % index):
            cell_out = tf.nn.max_pool(input_X, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'VALID')
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class Dropout(object):
    def __init__(self, input_X, is_training, index = 0):
        cell_out = tf.layers.dropout(inputs = input_X, rate = 0.5, training = is_training, name = 'Dropout_%d' % index)
        self.cell_out = cell_out

    def output(self):
        return self.cell_out


'''
First_Layer: def __init__(self, input_X, out_size, is_training=None, index = 0)
Residual_Block: def __init__(self, input_X, out_size, strides, is_training=None, index = 0)
Residual_Block_First: def __init__(self, input_X, out_size, strides, is_training=None, index = 0)
'''
def Network(input_X, input_Y, is_training, out_size):

    Layer_1 = First_Layer(input_X,
                          out_size = 64,
                          is_training = is_training,
                          index = 1)

    Layer_2_F = Residual_Block(Layer_1.output(),
                               out_size = 64,
                               strides = 1,
                               is_training = is_training,
                               index = 2)

    Layer_2_X = Dropout(Layer_2_F.output(),
                        is_training = is_training,
                        index = 1)

    Layer_2_S = Residual_Block(Layer_2_X.output(),
                               out_size = 64,
                               strides = 1,
                               is_training = is_training,
                               index = 3)

    Layer_2_D = Dropout(Layer_2_S.output(),
                        is_training = is_training,
                        index = 2)

    Layer_3_F = Residual_Block_First(Layer_2_D.output(),
                                     out_size = 128,
                                     strides = 2,
                                     is_training = is_training,
                                     index = 4)

    Layer_3_X = Dropout(Layer_3_F.output(),
                        is_training=is_training,
                        index=3)

    Layer_3_S = Residual_Block(Layer_3_X.output(),
                               out_size = 128,
                               strides = 1,
                               is_training = is_training,
                               index = 5)

    Layer_3_D = Dropout(Layer_3_S.output(),
                        is_training=is_training,
                        index=4)

    Layer_4_F = Residual_Block_First(Layer_3_D.output(),
                                     out_size = 256,
                                     strides = 2,
                                     is_training = is_training,
                                     index = 6)

    Layer_4_X = Dropout(Layer_4_F.output(),
                        is_training=is_training,
                        index=5)

    Layer_4_S = Residual_Block(Layer_4_X.output(),
                               out_size = 256,
                               strides = 1,
                               is_training = is_training,
                               index = 7)

    Layer_4_D = Dropout(Layer_4_S.output(),
                        is_training=is_training,
                        index=6)

    Layer_5_F = Residual_Block_First(Layer_4_D.output(),
                                     out_size = 512,
                                     strides = 2,
                                     is_training = is_training,
                                     index = 8)

    Layer_5_X = Dropout(Layer_5_F.output(),
                        is_training=is_training,
                        index=7)

    Layer_5_S = Residual_Block(Layer_5_X.output(),
                               out_size = 512,
                               strides = 1,
                               is_training = is_training,
                               index = 9)

    Layer_5_D = Dropout(Layer_5_S.output(),
                        is_training=is_training,
                        index=8)

    Layer_6 = MaxPool_Layer(Layer_5_D.output(),
                            index = 10)

    Layer_7 = FC(tf.reshape(Layer_6.output(), [-1,512]),
                 in_size = 512,
                 out_size = 10,
                 index = 11)

    output = Layer_7.output()

    # acc
    with tf.name_scope('error'):
        preds = tf.argmax(output, axis = 1)
        error = tf.count_nonzero(preds - input_Y, name = 'error')

    # loss
    with tf.name_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=input_Y)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)

    return loss, error





























