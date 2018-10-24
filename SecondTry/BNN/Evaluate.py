'''
Created on Sep/26/2018
@author: Bryce Xu
'''

import tensorflow as tf

def evaluate(output, input_Y):
    with tf.name_scope('evaluate'):
        prediction = tf.argmax(output, axis = 1)
        error = tf.count_nonzero(prediction - input_Y, name = 'error')
        tf.summary.scalar('error', error)
    return error