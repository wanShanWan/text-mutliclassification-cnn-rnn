# /usr/bin/python
# encoding : utf-8

"""
@ author : Wanshan
@ desc : Run TextCNN model(tensorflow)
"""

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from models.TextCNN_tf.model import TextCNNConfig, TextCNN
from preprocess import data_process, batch_iter, label_category


tensorboard_dir = './tensorboard_tectCNN/'
saver_dir = './checkpoint/'
train_dir = '../../data/train.txt'
valid_dir = '../../data/valid.txt'


def get_time_dif(start_time):
    """get time diff from 'start_time' to 'current' """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(sess, x_, y_):
    """Evaluate model in some data, return loss and acc"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.input_x: x_batch,
                     model.input_y: y_batch,
                     model.dropout_keep_prob: 1.0}
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(model):
    """Train model: split train data, train model, model save and result print"""

    # config tensor board and summary
    print('Configuring TensorBoard and Saver ...')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(tensorboard_dir)

    # config graph-saver
    saver = tf.train.Saver()
    if not os.path.exists(saver_dir):
        os.mkdir(saver_dir)

    # Loading trianing data and validation data
    print('Loading trianing data and validation data ...')
    start_time = time.time()
    x_train, y_train = data_process(train_dir, config.max_length)
    x_valid, y_valid = data_process(valid_dir, config.max_length)
    time_dif = get_time_dif(start_time)
    print('Loading data ok!')
    print('Time usage: %f' % time_dif)

    # Create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    summary_writer.add_graph(session.graph)

    # Some variables about training
    total_batch = 0
    best_val_acc = 0.
    last_improved = 0.
    early_stop_batch = 1000

    print('Training and evaluating ...')
    start_time = time.time()
    is_early_stop = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)

        for x_batch, y_batch in batch_train:
            feed_dict = {model.input_x: x_batch,
                         model.input_y: y_batch,
                         model.dropout_keep_prob: config.dropout_keep_prob}

            # Every saver_epochs, save summary
            if total_batch % config.save_per_batch == 0:
                graph = session.run(merged_summary, feed_dict=feed_dict)
                summary_writer.add_summary(graph, total_batch)

            # Print result
            if total_batch % config.print_per_batch == 0:
                feed_dict[model.dropout_keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                print()
                loss_val, acc_val = evaluate(session, x_valid, y_valid)

                # save best model by acc
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=saver_dir)
                    improved_str = '//improved'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            # Optimizer model
            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

            # Early stop
            if total_batch - last_improved > early_stop_batch:
                print("No optimization for a long time, auto-stopping...")
                is_early_stop = True
                break
        if is_early_stop:
            break


def inference(model):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = data_process(valid_dir, config.seq_length)

    # Create session and restore model checkpoint
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=saver_dir)

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    # Generate data with batch
    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # Evaluate
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=label_category))

    # Print confusion matrix
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'inference']:
        raise ValueError('usage: python run_cnn.py [train / inference]')

    print('Loading CNN model config ...')
    config = TextCNNConfig()

    model = TextCNN(max_length=config.max_length,
                    vocab_size=config.vocab_size,
                    embedding_size=config.embedding_dim,
                    filter_size=config.filter_szie,
                    num_filter=config.num_filter,
                    num_class=config.num_class)

    if sys.argv[1] == 'train':
        train(model)
    else:
        inference()



