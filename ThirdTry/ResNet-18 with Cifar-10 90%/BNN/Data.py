'''
Created on Oct/15/2018
@author: Bryce Xu
'''

import numpy as np
import platform
import os
from six.moves import cPickle as pickle


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
                yield x[a:b, :], y[a:b]
            # Shuffle the data
            else:
                batch_count = 0
                mask = np.arange(self.num_samples)
                np.random.shuffle(mask)
                x = x[mask]
                y = y[mask]

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


