# /usr/bin/python
# encoding : utf-8

"""
@ author : Wanshan
@ desc : Model of TextCNN, keras
"""

import argparse

import numpy as np
import pickle
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Activation, BatchNormalization,\
                         Dropout, Flatten, Dense
from keras import optimizers
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as bk


class TextCNNConfig(object):
    """Configuration of text CNN, Keras."""

    def __init__(self):
        parser = argparse.ArgumentParser()

        # param for text CNN
        parser.add_argument('--save_path', default='/', help='path to save model')

        # input param: max_length and batch size
        parser.add_argument('--max_length', type=int, default=400, help='max length of sentence')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')

        # embedding layer param
        parser.add_argument('--vocab_size', type=int, default=20000, help='size of vocab size')
        parser.add_argument('--embedding_size', type=int, default=128, help='set size of word embedding')
        parser.add_argument('--embedding_mode', type=bool, default=False, help='using pre_train weight matrix or not')

        # convolution layer param
        parser.add_argument('--kernel_size', type=list, default=[3, 4, 5], help='kernel size of different convolution')
        parser.add_argument('--kernel', type=list, default=[256, 128], help='kernel for different convolution')

        # output layer
        parser.add_argument('--num_class', type=int, default=202, help='number of classes')

        parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate of model')
        parser.add_argument('--num_epoch', type=int, default=25, help='num of maximum epoch')
        self.args = parser.parse_args()


class TextCNN(object):
    """
    Text CNN for text classification.
    Uses as embedding layer, followed different size convolution layer, max-pooling, fully link layer.
    """

    def __init__(self, config, weight_matrix=None):
        self.config = config

        # Input feed
        main_input = Input(shape=(config.max_length,), dtype='float64')

        # Embedding layer
        if config.embedding_mode:
            embedding = Embedding(config.vocab_size + 1,
                                  config.embedding_size,
                                  input_length=config.max_length)
        else:
            embedding = Embedding(config.vocab_size + 1,
                                  config.embedding_size,
                                  input_length=config.max_length,
                                  weights=weight_matrix,
                                  trainable=False)
        embed = embedding(main_input)

        # Convolution for different kernel size
        diff_convolution = []
        for size in config.kernel_size:
            convolution = embed
            for kernel in config.kernel:
                con_1oop = Convolution1D(kernel, size, padding='same')(convolution)
                bn_1oop = BatchNormalization()(con_1oop)
                act_1oop = Activation('relu')(bn_1oop)
                convolution = act_1oop
            pool = MaxPooling1D(pool_size=4)(convolution)
            dropout = Dropout(0.5)(pool)
            diff_convolution.append(dropout)

        # Concat
        concat = concatenate([x for x in diff_convolution], axis=-1)

        # Flatten
        flat = Flatten()(concat)
        drop = Dropout(0.5)(flat)

        # Two fully link layer
        fc_1 = Dense(512)(drop)
        bn_1 = BatchNormalization()(fc_1)
        drop_1 = Dropout(0.5)(bn_1)
        fc_2 = Dropout(256)(drop_1)
        bn_2 = BatchNormalization()(fc_2)
        drop_2 = Dropout(0.5)(bn_2)

        # Output layer
        main_output = Dense(config.num_class, activation='sigmoid')(drop_2)

        self.model = Model(inputs=main_input, outputs=main_output)
        self.model.summary()


if __name__ == '__main__':
    config = TextCNNConfig()
    model = TextCNN(config.args)