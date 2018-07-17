# /usr/bin/python
# encoding : utf-8

"""
@ author : Wanshan
@ desc :
"""

import tensorflow as tf
import numpy as np


class TextCNNConfig(object):
    """Config about text CNN"""

    max_length = 100  # maximum length of input layer
    embedding_dim = 256  # embedding size of word or char
    vocab_size = 2000  # length of vocabulary
    w2v_model = False  # if pre_train embedding or not

    filter_szie = [3, 4, 5]  # different convolution size
    num_filter = 128  # number of filter in each convolution

    num_class = 10  # number of classification classes

    dropout_keep_prob = 0.5  # dropout rate
    learning_rate = 1e-3  # learning rate

    batch_size = 128  # batch size
    num_epochs = 25  # total epoch of training

    print_per_batch = 10  # number of batch diff, print result
    save_per_batch = 10  # number of batch diff: save result


class TextCNN(object):
    """
    Text CNN for text classification.
    Uses as embedding layer, followed different size convolution layer, max-pooling, fully link layer.
    """

    def __init__(self, max_length, vocab_size, embedding_size, filter_size, num_filter, num_class, w2v_model=None,
                 learning_rate=0.001, l2_reg=0.0):

        # Input placeholder
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss
        l2_regular = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            if w2v_model is None:
                self.weight = tf.Variable(tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),
                                          name='word_embedding')
            else:
                self.weight = tf.get_variable('word_embedding', initializer=w2v_model.vectors.astype(np.float32))

            self.embedded_chars = tf.nn.embedding_lookup(self.weight, self.input_x)

        # Different size of convolution
        pooled_outputs = []
        for size in filter_size:
            with tf.name_scope('conv_maxpool-%s' % size):
                # Convolution layer

                conv_loop = tf.layers.conv1d(self.embedded_chars, num_filter, size, strides=1, padding='same')
                pooling_loop = tf.layers.max_pooling1d(conv_loop, 3, strides=1, padding='same')
                actv_loop = tf.nn.relu(pooling_loop)
                print(actv_loop.shape)

                pooled_outputs.append(actv_loop)

        # Concat all and fl
        self.concat = tf.concat(pooled_outputs, axis=-1)
        self.flat = tf.layers.flatten(self.concat, name='flatten')
        print(self.concat)

        # Two fully link layer and output
        with tf.name_scope("fully_link"):
            fc_1 = tf.layers.dense(self.flat, 512, name='fc1')
            print(fc_1.shape)
            dropout_1 = tf.layers.dropout(fc_1, self.dropout_keep_prob)
            actv_1 = tf.nn.relu(dropout_1)

            fc_2 = tf.layers.dense(actv_1, 256, name='fc2')
            print(fc_2.shape)
            dropout_2 = tf.layers.dropout(fc_2, self.dropout_keep_prob)
            actv_2 = tf.nn.relu(dropout_2)

            # output_layer
            self.logits = tf.layers.dense(actv_2, num_class, name='output')
            print(self.logits.shape)
            self.y_predict = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            # Loss compute: cross entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # Optimizer
            self.optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # Accuracy
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_predict)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    # initial model config
    text_cnn_config = TextCNNConfig()

    # test of model initial
    model = TextCNN(max_length=text_cnn_config.max_length,
                    vocab_size=text_cnn_config.vocab_size,
                    embedding_size=text_cnn_config.embedding_dim,
                    filter_size=text_cnn_config.filter_szie,
                    num_filter=text_cnn_config.num_filter,
                    num_class=text_cnn_config.num_class)


