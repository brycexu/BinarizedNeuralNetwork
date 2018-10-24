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
    with tf.name_scope('Binarize'):
        return tf.sign(input_X)

class CNN_Layer_First(object):
    def __init__(self, input_X, in_size, out_size, is_training = None, index = 0):
        with tf.name_scope('CNN_FIRST_%d' % index):
            with tf.variable_scope('CNN_Weights_%d' % index):
                W_shape = [3,3,in_size,out_size]
                Weight = tf.get_variable(name = 'CNN_FIRST_Weights_%d' % index, shape = W_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
                # Weight = tf.clip_by_value(Weight,-1,1);
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight
                # bias_shape = [out_size]
                # bias = tf.get_variable(name = 'CNN_FIRST_Bias_%d' % index, shape = bias_shape, initializer=tf.zeros_initializer)
                # self.bias = bias
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
                # Weight = tf.clip_by_value(Weight, -1, 1);
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight
                # bias_shape = [out_size]
                # bias = tf.get_variable(name='CNN_Layer_Bias_%d' % index, shape=bias_shape, initializer=tf.zeros_initializer)
                # self.bias = bias
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
                # Weight = tf.clip_by_value(Weight, -1, 1);
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight
                bias_shape = [out_size]
                # The inputs should be binarized now
                # bias = tf.get_variable(name = 'FC_Layer_Bias_%d' % index, shape = bias_shape, initializer = tf.contrib.layers.xavier_initializer())
                # self.bias = bias
                cell_out = tf.matmul(input_X, self.bin_Weight)
                self.cell_out = cell_out

                tf.summary.histogram('FC_Layer/{}/Weights'.format(index), self.Weight)
                tf.summary.histogram('FC_Layer/{}/bin_Weights'.format(index), self.bin_Weight)
                # tf.summary.histogram('FC_Layer/{}/Bias'.format(index), self.bias)

    def output(self):
        return self.cell_out

class FC_Layer_Last(object):
    def __init__(self, input_X, in_size, out_size, is_training = None, index = 0):
        with tf.name_scope('FC_Layer_%d' % index):
            with tf.variable_scope('FC_Weights_%d' % index):
                W_shape = [in_size, out_size]
                Weight = tf.get_variable(name='FC_Layer_Weights_%d' % index, shape=W_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
                # Weight = tf.clip_by_value(Weight, -1, 1);
                self.Weight = Weight
                bin_Weight = Binarize(Weight)
                self.bin_Weight = bin_Weight
                bias_shape = [out_size]
                # The inputs should not be binarized now
                bias = tf.get_variable(name='FC_Layer_Bias_%d' % index, shape=bias_shape,
                                        initializer=tf.contrib.layers.xavier_initializer())
                self.bias = bias
                cell_out = tf.add(tf.matmul(input_X, self.bin_Weight),bias)
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

    BN_Layer_3 = BN_Layer_CNN(input_X = CNN_Layer_3.output(),
                              is_training = is_training,
                              index = 3)

    HardTanh_Layer_3 = HardTanh_Layer(input_X = BN_Layer_3.output(),
                                      index = 3)

    CNN_Layer_4 = CNN_Layer(input_X = HardTanh_Layer_3.output(),
                            in_size = 256,
                            out_size = 256,
                            is_training = is_training,
                            index = 4)

    MaxPool_Layer_4 = MaxPool_Layer(input_X = CNN_Layer_4.output(),
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

    BN_Layer_5 = BN_Layer_CNN(input_X = CNN_Layer_5.output(),
                              is_training = is_training,
                              index = 5)

    HardTanh_Layer_5 = HardTanh_Layer(input_X = BN_Layer_5.output(),
                                      index = 5)

    CNN_Layer_6 = CNN_Layer(HardTanh_Layer_5.output(),
                            in_size = 512,
                            out_size = 512,
                            is_training = is_training,
                            index = 6)

    MaxPool_Layer_6 = MaxPool_Layer(input_X = CNN_Layer_6.output(),
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

    BN_Layer_7 = BN_Layer_FC(input_X = FC_Layer_7.output(),
                                is_training = is_training,
                                index = 7)

    HardTanh_Layer_7 = HardTanh_Layer(input_X = BN_Layer_7.output(),
                                      index = 7)

    FC_Layer_8 = FC_Layer(input_X = HardTanh_Layer_7.output(),
                          in_size = 1024,
                          out_size = 1024,
                          is_training = is_training,
                          index = 8)

    BN_Layer_8 = BN_Layer_FC(input_X = FC_Layer_8.output(),
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
        label = tf.one_hot(input_Y, out_size)
        loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(label, output)))
        tf.summary.scalar('loss', loss)




    def train_weight(loss = loss, CNN_Layer_First_1 = CNN_Layer_First_1, CNN_Layer_2 = CNN_Layer_2,
                     CNN_Layer_3 = CNN_Layer_3, CNN_Layer_4 = CNN_Layer_4, CNN_Layer_5 = CNN_Layer_5,
                     CNN_Layer_6 = CNN_Layer_6, FC_Layer_7 = FC_Layer_7, FC_Layer_8 = FC_Layer_8, FC_Layer_9_Last = FC_Layer_9):
        grad_Wb1, grad_Wb2, grad_Wb3, grad_Wb4, grad_Wb5, grad_Wb6, grad_Wb7, grad_Wb8, grad_Wb9 = tf.gradients(ys = loss,
                 xs = [CNN_Layer_First_1.bin_Weight, CNN_Layer_2.bin_Weight, CNN_Layer_3.bin_Weight, CNN_Layer_4.bin_Weight,
                      CNN_Layer_5.bin_Weight, CNN_Layer_6.bin_Weight, FC_Layer_7.bin_Weight, FC_Layer_8.bin_Weight, FC_Layer_9_Last.bin_Weight])
        '''
        grad_b1, grad_b2, grad_b3, grad_b4, grad_b5, grad_b6, grad_b7, grad_b8, grad_b9 = tf.gradients(ys = loss,
                 xs = [CNN_Layer_First_1.bias, CNN_Layer_2.bias, CNN_Layer_3.bias, CNN_Layer_4.bias, CNN_Layer_5.bias,
                       CNN_Layer_6.bias, FC_Layer_7.bias, FC_Layer_8.bias, FC_Layer_9_Last.bias])
        '''
        grad_b9 = tf.gradients(ys = loss, xs = FC_Layer_9_Last.bias)
        return (grad_Wb1, grad_Wb2, grad_Wb3, grad_Wb4, grad_Wb5, grad_Wb6, grad_Wb7, grad_Wb8, grad_Wb9, grad_b9)
                #grad_b1, grad_b2, grad_b3, grad_b4, grad_b5, grad_b6, grad_b7, grad_b8, grad_b9)

    def val_test_weight():
        return (0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
                #0., 0., 0., 0., 0., 0., 0., 0.)

    grad_update = tf.cond(is_training, lambda:train_weight(),lambda:val_test_weight())

    new_w1 = CNN_Layer_First_1.Weight.assign(tf.keras.backend.clip(CNN_Layer_First_1.Weight - lr * grad_update[0], -1.0, 1.0))
    new_w2 = CNN_Layer_2.Weight.assign(tf.keras.backend.clip(CNN_Layer_2.Weight - lr * grad_update[1], -1.0, 1.0))
    new_w3 = CNN_Layer_3.Weight.assign(tf.keras.backend.clip(CNN_Layer_3.Weight - lr * grad_update[2], -1.0, 1.0))
    new_w4 = CNN_Layer_4.Weight.assign(tf.keras.backend.clip(CNN_Layer_4.Weight - lr * grad_update[3], -1.0, 1.0))
    new_w5 = CNN_Layer_5.Weight.assign(tf.keras.backend.clip(CNN_Layer_5.Weight - lr * grad_update[4], -1.0, 1.0))
    new_w6 = CNN_Layer_6.Weight.assign(tf.keras.backend.clip(CNN_Layer_6.Weight - lr * grad_update[5], -1.0, 1.0))
    new_w7 = FC_Layer_7.Weight.assign(tf.keras.backend.clip(FC_Layer_7.Weight - lr * grad_update[6], -1.0, 1.0))
    new_w8 = FC_Layer_8.Weight.assign(tf.keras.backend.clip(FC_Layer_8.Weight - lr * grad_update[7], -1.0, 1.0))
    new_w9 = FC_Layer_9.Weight.assign(tf.keras.backend.clip(FC_Layer_9.Weight - lr * grad_update[8], -1.0, 1.0))

    '''
    new_b1 = CNN_Layer_First_1.bias.assign(CNN_Layer_First_1.bias - lr * grad_update[9])
    new_b2 = CNN_Layer_2.bias.assign(CNN_Layer_2.bias - lr * grad_update[10])
    new_b3 = CNN_Layer_3.bias.assign(CNN_Layer_3.bias - lr * grad_update[11])
    new_b4 = CNN_Layer_4.bias.assign(CNN_Layer_4.bias - lr * grad_update[12])
    new_b5 = CNN_Layer_5.bias.assign(CNN_Layer_5.bias - lr * grad_update[13])
    new_b6 = CNN_Layer_6.bias.assign(CNN_Layer_6.bias - lr * grad_update[14])
    new_b7 = FC_Layer_7.bias.assign(FC_Layer_7.bias - lr * grad_update[15])
    new_b8 = FC_Layer_8.bias.assign(FC_Layer_8.bias - lr * grad_update[16])
    '''
    new_b9 = FC_Layer_9.bias.assign(FC_Layer_9.bias - lr * grad_update[9])

    return output, loss, (new_b9, new_w1, new_w2, new_w3, new_w4, new_w5, new_w6, new_w7, new_w8, new_w9)

    '''
    return output, loss
    '''









































