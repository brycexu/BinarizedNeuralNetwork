'''
Created on Sep/15/2018
@author: Bryce Xu
'''

import tensorflow as tf
import numpy as np
import time

# Shuffle the data
# Prepare the batches
class DataGenerator(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_samples = x.shape[0]
    
    def dataGenerator(self, batch_size):
        x = self.x
        y = self.y
        num_batch = self.num_samples // batch_size
        batch_count = 0
        while 1:
            # Create batches
            if batch_count < num_batch:
                a = batch_count * batch_size
                b = (batch_count + 1) * batch_size
                batch_count += 1
                yield x[a:b, : ], y[a:b]
            # Shuffle the data
            else:
                batch_count = 0
                mask = np.arange(self.num_samples)
                np.random.shuffle(mask)
                x = x[mask]
                y = y[mask]

# BN Layer for CNN
class NormLayer_CNN(object):
    def __init__(self, input_X, is_training = None, activation_function = tf.nn.relu, index = 0):
        '''
        def batch_norm_training():
            batch_mean, batch_var = tf.nn.moments(input_X, [0,1,2], keep_dims = False)
            decay = 0.99
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(input_X, batch_mean, batch_var, beta, gamma, epsilon)

        def batch_norm_inference():
            return tf.nn.batch_normalization(input_X, pop_mean, pop_variance, beta, gamma, epsilon)

        with tf.variable_scope('BN_FC%d' % index):
            gamma = tf.Variable(tf.ones([input_X.shape[3]]))
            beta = tf.Variable(tf.zeros([input_X.shape[3]]))
            pop_mean = tf.Variable(tf.zeros([input_X.shape[3]]), trainable = False)
            pop_variance = tf.Variable(tf.ones([input_X.shape[3]]), trainable = False)
            epsilon = 1e-4
            cell_out = tf.cond(is_training, lambda:batch_norm_training(), lambda:batch_norm_inference())
            cell_out = activation_function(cell_out)
            self.cell_out = cell_out
        '''
        with tf.variable_scope('BN_CNN%d' % index):
            batch_mean, batch_var = tf.nn.moments(input_X, [0,1,2], keep_dims = True)
            shift = tf.Variable(tf.zeros([input_X.shape[3]]))
            scale = tf.Variable(tf.ones([input_X.shape[3]]))
            epsilon = 1e-4
            cell_out = tf.nn.batch_normalization(input_X, batch_mean, batch_var, shift, scale, epsilon)
            cell_out = activation_function(cell_out)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

# BN Layer for FC
class NormLayer_FC(object):
    def __init__(self, input_X, is_training = None, activation_function = tf.nn.relu, index = 0):
        with tf.variable_scope('BN_FC%d' % index):
            cell_out = tf.contrib.layers.batch_norm(input_X, decay = 0.99, is_training = is_training, epsilon = 1e-4)
            cell_out = activation_function(cell_out)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

# MaxPooling Layer for CNN
class MaxPoolLayer(object):
    def __init__(self, input_X, index = 0):
        with tf.variable_scope('MP_CNN%d' % index):
            cell_out = tf.nn.max_pool(input_X, ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

# FC Layer
class FCLayer(object):
    def __init__(self, input_X, in_size, out_size, binary = False, is_training = None, index = 0):
        with tf.variable_scope('FC_%d' % index):
            with tf.name_scope('FC_Weights'):
                W_shape = [in_size, out_size]
                Weight = tf.get_variable(name = 'FC_Weights_%d' % index, shape = W_shape, initializer = tf.contrib.layers.xavier_initializer())
                self.Weight = Weight
                b_shape = [out_size]
                bias = tf.get_variable(name = 'FC_Bias_%d' % index, shape = b_shape, initializer = tf.contrib.layers.xavier_initializer())
                self.bias = bias

            if binary:
                Wb = tf.cond(is_training, lambda: binarization(Weight, binary = binary), lambda: Weight)
            else:
                Wb = binarization(Weight, binary = binary)

            self.Wb = Wb
            cell_out = tf.add(tf.matmul(input_X, self.Wb) , bias)
            self.cell_out = cell_out
            tf.summary.histogram('FC_Layer/{}/Weights'.format(index), self.Weight)
            tf.summary.histogram('FC_Layer/{}/Bias'.format(index), self.bias)

    def output(self):
        return self.cell_out

# CNN Layer
class CNNLayer(object):
    def __init__(self, input_X, in_size, out_size, binary = False, is_training = None, index = 0):
        with tf.variable_scope('CNN_%d' % index):
            with tf.name_scope('CNN_Weights'):
                W_shape = [3, 3, in_size, out_size]
                Weight = tf.get_variable(name='CNN_Weights_%d' % index, shape=W_shape, initializer=tf.contrib.layers.xavier_initializer())
                self.Weight = Weight
                b_shape = [out_size]
                bias = tf.get_variable(name = 'CNN_Bias_%d' % index, shape = b_shape, initializer = tf.contrib.layers.xavier_initializer())
                self.bias = bias

            if binary:
                Wb = tf.cond(is_training, lambda: binarization(Weight, binary = binary), lambda: Weight)
            else:
                Wb = binarization(Weight, binary = binary)

            self.Wb = Wb
            cell_out = conv2d(input_X, self.Wb) + bias
            self.cell_out = cell_out
            tf.summary.histogram('CNN_Layer/{}/Weights'.format(index), self.Weight)
            tf.summary.histogram('CNN_Layer/{}/Bias'.format(index), self.bias)

    def output(self):
        return self.cell_out

# Convolution for CNN Layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

# Hard Sigmoid for clipping the data
def hardSigmoid(x):
    return tf.keras.backend.clip((x+1.0)/2.0, 0.0, 1.0)

# Binarize the data
def binarization(W, binary = False):
    if binary:
        Wb = hardSigmoid(W)
        Wb = tf.round(Wb)
        Wb = Wb * 2 - 1
    else:
        Wb = W
    return Wb

# Evaluate the output
def evaluate(output, input_Y):
    with tf.name_scope('evaluate'):
        prediction = tf.argmax(output, axis = 1)
        error = tf.count_nonzero(prediction - input_Y, name = 'error')
        tf.summary.scalar('error', error)
    return error

# Build the network
def Network(input_X, input_Y, is_training, is_binary, out_size, fc_units, cnn_units, lr):

    # LeNet

    # Input - CNN - BN - Relu - MaxPool - CNN - BN - Relu - MaxPool - FC - BN - FC - BN - Output

    # CNN Layer 1:
    #   input: 28*28*1
    #   kernel: 3x3x32
    #   stride: 1
    #   padding: SAME
    #   output: 28*28*32
    # Norm Layer 1:
    #   input: 28*28*32
    #   output: 28*28*32
    # MaxPool Layer 1:
    #   input: 28*28*32
    #   output: 14*14*32
    # CNN Layer 2:
    #   input: 14*14*32
    #   kernel: 3x3x64
    #   stride: 1
    #   padding: SAME
    #   output: 14*14*64
    # Norm Layer 2:
    #   input: 14*14*64
    #   output: 14*14*64
    # MaxPool Layer 2:
    #   input: 14*14*64
    #   output: 7*7*64
    # FC Layer 3:
    #   input: 3136
    #   output: 1024
    # Norm Layer 3:
    #   input: 1024
    #   output: 1024
    # FC Layer 4:
    #   input:1024
    #   output: 10
    # Norm Layer 4:
    #   input:10
    #   output:10

    CNN_Layer_1 = CNNLayer(input_X = input_X, # 28*28*1
                           in_size = 1,
                           out_size = cnn_units[0], # 32
                           binary = is_binary,
                           is_training = is_training,
                           index = 0) # 28*28*32

    Norm_Layer_1 = NormLayer_CNN(input_X = CNN_Layer_1.output(),
                             is_training = is_training,
                             activation_function = tf.nn.relu,
                             index = 0)

    MAXPOOL_Layer_1 = MaxPoolLayer(input_X = Norm_Layer_1.output(),
                                   index = 0)


    CNN_Layer_2 = CNNLayer(input_X = MAXPOOL_Layer_1.output(), # 14*14*32
                           in_size = cnn_units[0], # 32
                           out_size = cnn_units[1], # 64
                           binary = is_binary,
                           is_training = is_training,
                           index = 1) # 14*14*64


    Norm_Layer_2 = NormLayer_CNN(input_X = CNN_Layer_2.output(),
                             is_training = is_training,
                             activation_function = tf.nn.relu,
                             index = 1)

    MAXPOOL_Layer_2 = MaxPoolLayer(input_X = Norm_Layer_2.output(),
                                   index = 1) # 7*7*64

    FC_Layer_3 = FCLayer(input_X = tf.reshape(MAXPOOL_Layer_2.output(),[-1,7*7*64]),
                         in_size = fc_units[0], # 7*7*64
                         out_size = fc_units[1], # 1024
                         binary = is_binary,
                         is_training = is_training,
                         index = 2)

    Norm_Layer_3 = NormLayer_FC(input_X = FC_Layer_3.output(),
                             is_training = is_training,
                             activation_function = tf.nn.relu,
                             index = 2)

    FC_Layer_4 = FCLayer(input_X = Norm_Layer_3.output(),
                         in_size = fc_units[1], # 1024
                         out_size = fc_units[2], # 10
                         binary = is_binary,
                         is_training = is_training,
                         index = 3)

    Norm_Layer_4 = NormLayer_FC(input_X = FC_Layer_4.output(),
                                is_training = is_training,
                                activation_function = tf.nn.relu,
                                index = 3)

    with tf.name_scope('loss'):
        output = Norm_Layer_4.output()
        label = tf.one_hot(input_Y, out_size)
        loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(label, output)))
        tf.summary.scalar('loss', loss)
        
    def train_weight(loss = loss, CNN_Layer_1 = CNN_Layer_1, CNN_Layer_2 = CNN_Layer_2, FC_Layer_3 = FC_Layer_3, FC_Layer_4 = FC_Layer_4):
        grad_Wb0, grad_Wb1, grad_Wb2, grad_Wb3 = tf.gradients(ys = loss, xs = [CNN_Layer_1.Wb, CNN_Layer_2.Wb, FC_Layer_3.Wb, FC_Layer_4.Wb])
        grad_b0, grad_b1, grad_b2, grad_b3 = tf.gradients(ys=loss, xs=[CNN_Layer_1.bias, CNN_Layer_2.bias, FC_Layer_3.bias, FC_Layer_4.bias])
        return (grad_Wb0, grad_Wb1, grad_Wb2, grad_Wb3, grad_b0, grad_b1, grad_b2, grad_b3)
    
    def val_test_weight():
        return (0., 0., 0., 0., 0., 0., 0., 0.)

    grad_update = tf.cond(is_training, lambda:train_weight(), lambda: val_test_weight())

    if is_binary:
        new_w0 = CNN_Layer_1.Weight.assign(tf.keras.backend.clip(CNN_Layer_1.Weight - lr * grad_update[0], -1.0, 1.0))
        new_w1 = CNN_Layer_2.Weight.assign(tf.keras.backend.clip(CNN_Layer_2.Weight - lr * grad_update[1], -1.0, 1.0))
        new_w2 = FC_Layer_3.Weight.assign(tf.keras.backend.clip(FC_Layer_3.Weight - lr * grad_update[2], -1.0, 1.0))
        new_w3 = FC_Layer_4.Weight.assign(tf.keras.backend.clip(FC_Layer_4.Weight - lr * grad_update[3], -1.0, 1.0))
    else:
        new_w0 = CNN_Layer_1.Weight.assign(CNN_Layer_1.Weight - lr * grad_update[0])
        new_w1 = CNN_Layer_2.Weight.assign(CNN_Layer_2.Weight - lr * grad_update[1])
        new_w2 = FC_Layer_3.Weight.assign(FC_Layer_3.Weight - lr * grad_update[2])
        new_w3 = FC_Layer_4.Weight.assign(FC_Layer_4.Weight - lr * grad_update[3])

    new_b0 = CNN_Layer_1.bias.assign(CNN_Layer_1.bias - lr * grad_update[4])
    new_b1 = CNN_Layer_2.bias.assign(CNN_Layer_2.bias - lr * grad_update[5])
    new_b2 = FC_Layer_3.bias.assign(FC_Layer_3.bias - lr * grad_update[6])
    new_b3 = FC_Layer_4.bias.assign(FC_Layer_4.bias - lr * grad_update[7])

    return output, loss, (new_b0, new_b1, new_b2, new_b3, new_w0, new_w1, new_w2, new_w3)

def training(X_train, Y_train, X_val, Y_val, X_test, Y_test, binary, cnn_units, fc_units, lr_start, lr_end, epoch, batch_size):
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape = [None, 784], dtype = tf.float32, name = 'xs')
        ys = tf.placeholder(shape = [None, ], dtype = tf.int64, name = 'ys')
        is_training = tf.placeholder(dtype = tf.bool, name = 'is_training')
        x_input = tf.reshape(xs,[-1,28,28,1])
        
        learning_rate = tf.Variable(lr_start, name = 'learning_rate')
        lr_decay = (lr_end / lr_start) ** (1 / epoch)
        lr_update = learning_rate.assign(learning_rate * lr_decay)
        
        train_data_generator = DataGenerator(X_train, Y_train)
        train_batch_generator = train_data_generator.dataGenerator(batch_size)
        iters = int(X_train.shape[0] / batch_size)
        print('number of batches for traning: {}'.format(iters))
        
        val_batch_size = 100
        val_data_generator = DataGenerator(X_val, Y_val)
        val_batch_generator = val_data_generator.dataGenerator(val_batch_size)
        print('data generator init')
        
        output, loss, _updates = Network(x_input, ys,
                                         is_training = is_training,
                                         is_binary= binary, 
                                         out_size = 10, 
                                         cnn_units = cnn_units, 
                                         fc_units = fc_units, 
                                         lr = learning_rate)
        
        errorate = evaluate(output, ys)
        best_acc = 0
        cur_model_name = 'mnist_{}'.format(int(time.time()))
        
        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer(), {is_training: False})
        
            for epc in range(epoch):
                print("epoch {} ".format(epc + 1))
                # Training
                train_eve_sum = 0
                loss_sum = 0
                for _ in range(iters):
                    train_batch_x, train_batch_y = next(train_batch_generator)
                    _, cur_loss, train_eve = sess.run([_updates, loss, errorate],
                                                      feed_dict={xs: train_batch_x, ys: train_batch_y, is_training: True})
                    train_eve_sum += np.sum(train_eve)
                    loss_sum += np.sum(cur_loss)
                train_acc = 100 - train_eve_sum * 100 / Y_train.shape[0]
                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))
                # Validation
                valid_eve_sum = 0
                for _ in range(Y_val.shape[0] // val_batch_size):
                    val_batch_x, val_batch_y = next(val_batch_generator)
                    valid_eve, merge_result = sess.run([errorate, merge],
                                                       feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})
                    valid_eve_sum += np.sum(valid_eve)
                valid_acc = 100 - valid_eve_sum * 100 / Y_val.shape[0]
                _lr = sess.run([lr_update])
                print('validation accuracy : {}%'.format(valid_acc))

                # Save the merge result summary
                writer.add_summary(merge_result, epc)

                # When achieve the best validation accuracy, we store the model paramters
                if valid_acc > best_acc:
                    print('* Best accuracy: {}%'.format(valid_acc))
                    best_acc = valid_acc
                    saver.save(sess, 'model/{}'.format(cur_model_name))
                print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))

            else:
                pass





















