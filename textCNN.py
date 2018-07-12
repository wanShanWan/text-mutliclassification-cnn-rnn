# /usr/bin/python
# encoding : utf-8

"""
@ author : Wanshan
@ desc :
"""

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    Text CNN for text classification.
    Uses as embedding layer, followed different size convolution layer, max-pooling, fully link layer.
    """

    def __init__(self, max_length, vocab_size, embedding_size, filter_size, num_filter,num_class, w2v_model=None,
                 l2_reg=0.0):

        # Input placeholder
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            if w2v_model is None:
                self.weight = tf.Variable(tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),
                                          name='word_embedding')
            else:
                self.weight = tf.get_variable('word_embedding', initializer=w2v_model.vectors.astype(np.float32))

            self.embedded_chars = tf.nn.embedding_lookup(self.w, self.input_x)
