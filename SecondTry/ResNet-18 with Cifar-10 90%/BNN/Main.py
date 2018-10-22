'''
Created on Oct/15/2018
@author: Bryce Xu
'''

import tensorflow as tf
from LoadCifar10 import load_cifar10

X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print("Number of training samples:", X_train.shape[0])
print("Number of validation samples:", X_val.shape[0])
print("Number of test samples:", X_test.shape[0])


from Training import training

tf.reset_default_graph()

test_pred = training(X_train, y_train, X_val, y_val,
                     lr_start = 0.1,
                     lr_end = 0.01,
                     epoch = 100,
                     batch_size = 128)
