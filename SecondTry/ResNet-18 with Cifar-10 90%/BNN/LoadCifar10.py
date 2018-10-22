# encoding: utf-8
'''
Created on Oct/15/2018
@author: Bryce Xu
'''

import numpy as np
from Data import load_CIFAR10


def load_cifar10(num_training=40000, num_validation=10000, num_test=10000):
    # CIFAT数据包位置
    cifat10_path = '/Users/XuXianda/Desktop/BNN_2/BNN/cifar-10-batches-py'
    # 导入数据
    X_train, y_train, X_test, y_test = load_CIFAR10(cifat10_path)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()
    # 重新处理数据,引入val
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # 一般化数据:subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test
