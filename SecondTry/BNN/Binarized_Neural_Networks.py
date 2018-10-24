'''
Created on Sep/26/2018
@author: Bryce Xu
'''

import tensorflow as tf

'''
AlexNet
Input
    Cifar-10:32x32x3
SpatialConvolution (Weight Only) 'VALID'
    [3,3,3,128]
    [1,1,1,1]
    32x32x3 -> 32x32x128
BatchNormalization
HardTanh
SpartialConvolution_1
    [3,3,128,128]
    [1,1,1,1]
    32x32x128 -> 32x32x128
MaxPooling
    [1,2,2,1] [1,2,2,1]
    32x32x128 -> 16x16x128
BatchNormalization
HardTanh
SpartialConvolution_2
    [3,3,128,256]
    [1,1,1,1]
    16x16x128 -> 16x16x256
BatchNormalization
HardTanh
SpartialConvolution_3
    [3,3,256,256]
    [1,1,1,1]
    16x16x256 -> 16x16x256
MaxPooling
    [1,2,2,1] [1,2,2,1] 
    16x16x256 -> 8x8x256
BatchNormalization
HardTanh
SpartialConvolution_4
    [3,3,256,512]
    [1,1,1,1]
    8x8x256 -> 8x8x512
BatchNormalization
HardTanh
SpartialConvolution_5
    [3,3,512,512]
    [1,1,1,1]
    8x8x512 -> 8x8x512
MaxPooling
    [1,2,2,1] [1,2,2,1] 
    8x8x512 -> 4x4x512
BatchNormalization
HardTanh
FC_1
    8192 -> 1024
BatchNormalization
HardTanh
FC_2
    1024 -> 1024
BatchNormalization
HardTanh
FC_3
    1024 -> 10
BatchNormalization
'''

def Binarize(input_X):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign":"Identity"}):
        return tf.sign(input_X)

class CNN_Layer_First(object):
    def __init__(self, input_X, in_size, out_size, is_training = None, index = 0):
        with tf.name_scope('CNN_FIRST_%d' % index):
            with tf.variable_scope('CNN_Weights_%d' % index):
                W_shape = [3,3,in_size,out_size]
                Weight = tf.get_variable(name = 'CNN_FIRST_Weights_%d' % index, shape = W_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
                Weight = tf.clip_by_value(Weight,-1,1)
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight
                cell_out = tf.nn.conv2d(input_X, bin_Weight, strides = [1,1,1,1], padding = "VALID")
                self.cell_out = cell_out

                tf.summary.histogram('CNN_Layer_FIRST/{}/Weights'.format(index),self.Weight)
                tf.summary.histogram('CNN_Layer_FIRST/{}/bin_Weights'.format(index),self.bin_Weight)
                # tf.summary.histogram('CNN_Layer_FIRST/{}/Bias'.format(index),self.bias)

    def output(self):
        return self.cell_out

class CNN_Layer(object):
    def __init__(self, input_X, in_size, out_size, is_training = None, index = 0):
        with tf.name_scope('CNN_Layer_%d' % index):
            with tf.variable_scope('CNN_Weights_%d' % index):
                W_shape = [3,3,in_size,out_size]
                Weight = tf.get_variable(name='CNN_Layer_Weights_%d' % index, shape=W_shape,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d())
                Weight = tf.clip_by_value(Weight, -1, 1)
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight

                # The inputs shoule be binarized now
                input_X = Binarize(input_X)
                cell_out = tf.nn.conv2d(input_X, bin_Weight, strides=[1, 1, 1, 1], padding="SAME")
                self.cell_out = cell_out

                tf.summary.histogram('CNN_Layer_FIRST/{}/Weights'.format(index), self.Weight)
                tf.summary.histogram('CNN_Layer_FIRST/{}/bin_Weights'.format(index), self.bin_Weight)
                # tf.summary.histogram('CNN_Layer_FIRST/{}/Bias'.format(index), self.bias)

    def output(self):
        return self.cell_out


class FC_Layer(object):
    def __init__(self, input_X, in_size, out_size, is_training = None, index = 0):
        with tf.name_scope('FC_Layer_%d' % index):
            with tf.variable_scope('FC_Weights_%d' % index):
                input_X = Binarize(input_X)
                W_shape = [in_size,out_size]
                Weight = tf.get_variable(name = 'FC_Layer_Weights_%d' % index, shape = W_shape, initializer = tf.contrib.layers.xavier_initializer())
                Weight = tf.clip_by_value(Weight, -1, 1)
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight
                cell_out = tf.matmul(input_X, bin_Weight)
                self.cell_out = cell_out

                tf.summary.histogram('FC_Layer/{}/Weights'.format(index), self.Weight)
                tf.summary.histogram('FC_Layer/{}/bin_Weights'.format(index), self.bin_Weight)
                # tf.summary.histogram('FC_Layer/{}/Bias'.format(index), self.bias)

    def output(self):
        return self.cell_out

class Dropout_FC(object):
    def __init__(self, input_X, is_training, index = 0):
        cell_out = tf.layers.dropout(inputs = input_X, rate = 0.3, training = is_training, name = 'Dropout_%d' % index)
        self.cell_out = cell_out

    def output(self):
        return self.cell_out

class Dropout_CNN(object):
    def __init__(self, input_X, is_training, index = 0):
        cell_out = tf.layers.dropout(inputs = input_X, rate = 0.1, training = is_training, name = 'Dropout_%d' % index)
        self.cell_out = cell_out

    def output(self):
        return self.cell_out

class FC_Layer_Last(object):
    def __init__(self, input_X, in_size, out_size, is_training = None, index = 0):
        with tf.name_scope('FC_Layer_%d' % index):
            with tf.variable_scope('FC_Weights_%d' % index):
                W_shape = [in_size, out_size]
                Weight = tf.get_variable(name='FC_Layer_Weights_%d' % index, shape=W_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
                # Weight = tf.clip_by_value(Weight, -1, 1)
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight
                # bias_shape = [out_size]
                # The inputs should not be binarized now

                cell_out = tf.matmul(input_X, Weight)
                self.cell_out = cell_out

                tf.summary.histogram('FC_Layer/{}/Weights'.format(index), self.Weight)
                tf.summary.histogram('FC_Layer/{}/bin_Weights'.format(index), self.bin_Weight)
                # tf.summary.histogram('FC_Layer/{}/Bias'.format(index), self.bias)

    def output(self):
        return self.cell_out


class BN_Layer_CNN(object):
    def __init__(self, input_X, is_training, index = 0):
        with tf.name_scope('BNN_Layer_%d' % index):
            with tf.variable_scope('BN_CNN_%d' % index):
                batch_mean, batch_var = tf.nn.moments(input_X, [0,1,2], keep_dims = True)
                shift = tf.Variable(tf.zeros([input_X.shape[3]]))
                scale = tf.Variable(tf.ones([input_X.shape[3]]))
                epsilon = 1e-4
                cell_out = tf.nn.batch_normalization(input_X, batch_mean, batch_var, shift, scale, epsilon)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out

class BN_Layer_FC(object):
    def __init__(self, input_X, is_training, index = 0):
        with tf.name_scope('BNN_Layer_%d' % index):
            with tf.variable_scope('BN_FC_%d' % index):
                cell_out = tf.contrib.layers.batch_norm(input_X, decay = 0.99, is_training = is_training, epsilon = 1e-4)
                self.cell_out = cell_out

    def output(self):
        return self.cell_out


class MaxPool_Layer(object):
    def __init__(self, input_X, index = 0):
        with tf.name_scope('MP_CNN_%d' % index):
            cell_out = tf.nn.max_pool(input_X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

class HardTanh_Layer(object):
    def __init__(self, input_X, index = 0):
        with tf.name_scope('HT_%d' % index):
            cell_out = tf.clip_by_value(input_X,-1,1)
            # cell_out = tf.nn.relu(input_X)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


def Network(input_X, input_Y, is_training, out_size, lr):

    CNN_Layer_First_1 = CNN_Layer_First(input_X = input_X,
                                        in_size = 3,
                                        out_size = 128,
                                        is_training = is_training,
                                        index = 1)

    BN_Layer_1 = BN_Layer_CNN(input_X = CNN_Layer_First_1.output(),
                              is_training = is_training,
                              index = 1)

    HardTanh_Layer_1 = HardTanh_Layer(input_X = BN_Layer_1.output(),
                                      index = 1)

    CNN_Layer_2 = CNN_Layer(input_X = HardTanh_Layer_1.output(),
                            in_size = 128,
                            out_size = 128,
                            is_training = is_training,
                            index = 2)

    MaxPool_Layer_2 = MaxPool_Layer(input_X = CNN_Layer_2.output(),
                                    index = 2)

    BN_Layer_2 = BN_Layer_CNN(input_X = MaxPool_Layer_2.output(),
                              is_training = is_training,
                              index = 2)

    HardTanh_Layer_2 = HardTanh_Layer(BN_Layer_2.output(),
                                      index = 2)

    CNN_Layer_3 = CNN_Layer(input_X = HardTanh_Layer_2.output(),
                            in_size = 128,
                            out_size = 256,
                            is_training = is_training,
                            index = 3)

    Dropout_Layer_3 = Dropout_CNN(input_X = CNN_Layer_3.output(),
                                  is_training = is_training,
                                  index = 3)

    BN_Layer_3 = BN_Layer_CNN(input_X = Dropout_Layer_3.output(),
                              is_training = is_training,
                              index = 3)

    HardTanh_Layer_3 = HardTanh_Layer(input_X = BN_Layer_3.output(),
                                      index = 3)

    CNN_Layer_4 = CNN_Layer(input_X = HardTanh_Layer_3.output(),
                            in_size = 256,
                            out_size = 256,
                            is_training = is_training,
                            index = 4)

    Dropout_Layer_4 = Dropout_CNN(input_X=CNN_Layer_4.output(),
                                  is_training=is_training,
                                  index=4)

    MaxPool_Layer_4 = MaxPool_Layer(input_X = Dropout_Layer_4.output(),
                                    index = 4)

    BN_Layer_4 = BN_Layer_CNN(input_X = MaxPool_Layer_4.output(),
                              is_training = is_training,
                              index = 4)

    HardTanh_Layer_4 = HardTanh_Layer(input_X = BN_Layer_4.output(),
                                      index = 4)

    CNN_Layer_5 = CNN_Layer(input_X = HardTanh_Layer_4.output(),
                            in_size = 256,
                            out_size = 512,
                            is_training = is_training,
                            index = 5)

    Dropout_Layer_5 = Dropout_CNN(input_X=CNN_Layer_5.output(),
                                  is_training=is_training,
                                  index=5)

    BN_Layer_5 = BN_Layer_CNN(input_X = Dropout_Layer_5.output(),
                              is_training = is_training,
                              index = 5)

    HardTanh_Layer_5 = HardTanh_Layer(input_X = BN_Layer_5.output(),
                                      index = 5)

    CNN_Layer_6 = CNN_Layer(HardTanh_Layer_5.output(),
                            in_size = 512,
                            out_size = 512,
                            is_training = is_training,
                            index = 6)

    Dropout_Layer_6 = Dropout_CNN(input_X=CNN_Layer_6.output(),
                                  is_training=is_training,
                                  index=6)

    MaxPool_Layer_6 = MaxPool_Layer(input_X = Dropout_Layer_6.output(),
                                    index = 6)

    BN_Layer_6 = BN_Layer_CNN(input_X = MaxPool_Layer_6.output(),
                              is_training = is_training,
                              index = 6)

    HardTanh_Layer_6 = HardTanh_Layer(input_X = BN_Layer_6.output(),
                                      index = 6)

    FC_Layer_7 = FC_Layer(input_X = tf.reshape(HardTanh_Layer_6.output(),[-1,4608]),
                          in_size = 4608,
                          out_size = 1024,
                          is_training = is_training,
                          index = 7)

    Dropout_Layer_7 = Dropout_FC(FC_Layer_7.output(),
                              is_training=is_training,
                              index=7)

    BN_Layer_7 = BN_Layer_FC(input_X = Dropout_Layer_7.output(),
                                is_training = is_training,
                                index = 7)

    HardTanh_Layer_7 = HardTanh_Layer(input_X = BN_Layer_7.output(),
                                      index = 7)

    FC_Layer_8 = FC_Layer(input_X = HardTanh_Layer_7.output(),
                          in_size = 1024,
                          out_size = 1024,
                          is_training = is_training,
                          index = 8)

    Dropout_Layer_8 = Dropout_FC(FC_Layer_8.output(),
                              is_training=is_training,
                              index=8)

    BN_Layer_8 = BN_Layer_FC(input_X = Dropout_Layer_8.output(),
                             is_training = is_training,
                             index = 8)

    HardTanh_Layer_8 = HardTanh_Layer(input_X = BN_Layer_8.output(),
                                      index = 8)

    FC_Layer_9 = FC_Layer_Last(input_X = HardTanh_Layer_8.output(),
                          in_size = 1024,
                          out_size = 10,
                          is_training = is_training,
                          index = 9)

    BN_Layer_9 = BN_Layer_FC(input_X = FC_Layer_9.output(),
                             is_training = is_training,
                             index = 9)

    with tf.name_scope('loss'):
        output = BN_Layer_9.output()
        # label = tf.one_hot(input_Y, out_size)
        # loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(label, output)))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=input_Y)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)



    return output, loss






































