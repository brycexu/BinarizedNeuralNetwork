'''
Created on Sep/15/2018
@author: Bryce Xu
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Prepare the data
mnist = input_data.read_data_sets('MNIST_data', one_hot = False, validation_size = 10000)
X_train = mnist.train.images
Y_train = mnist.train.labels
X_val = mnist.validation.images
Y_val = mnist.validation.labels
X_test = mnist.test.images
Y_test = mnist.test.labels
print("Number of training samples:" , mnist.train.num_examples)
print("Number of validation samples:" , mnist.validation.num_examples)
print("Number of test samples:" , mnist.test.num_examples)

from Binary_Neural_Network import training

# Training
tf.reset_default_graph()
test_pred = training(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                     binary = True,
                     cnn_units = [32, 64],
                     fc_units = [3136, 1024, 10],
                     lr_start = 20.0,
                     lr_end = 10.0,
                     epoch = 100,
                     batch_size = 60)






