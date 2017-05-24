#!/usr/bin/env python3
# coding: utf-8

"""
卷积神经网络实现文本情感分类
usage: `python conv_nlp.py train_file test_file`
"""

from collections import Counter
import sys
import time

import numpy as np
import tensorflow as tf

from data_reader import DOC_LENGTH, build_vocab, generate_inputs_and_labels


word_embed_size = 30
filter_num = 30
window_size = 3
rounds = 500  # 训练时遍历样本集的轮数
print_cost_every = 10  # 训练中计算并打印 cost 的间隔数
batch_size = 100
learning_rate = 1


def generate_placeholders():
    inputs = tf.placeholder(tf.int32, shape=[None, DOC_LENGTH], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    return inputs, labels


def build_convnet(vocab, inputs, labels):
    W = tf.Variable(tf.random_uniform([len(vocab), word_embed_size], -1.0, 1.0), name="W")
    embeds = tf.nn.embedding_lookup(W, inputs)
    embeds_expand = tf.expand_dims(embeds, -1)  # 扩展维度, 适应 conv2d 的参数

    # 构建 conv 层 + pool 层
    with tf.name_scope("conv-maxpool"):
        filter_shape = [window_size, word_embed_size, 1, filter_num]

        # 卷积层参数
        W = tf.Variable(tf.random_uniform(filter_shape, -1.0, 1.0), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[filter_num]), name="b")

        conv = tf.nn.conv2d(
            embeds_expand, W, strides=[1, 1, 1, 1],
            padding="VALID", name="conv")
        conv_hidden = tf.nn.tanh(tf.add(conv, b), name="tanh")

        # pool 层
        pool = tf.nn.max_pool(
            conv_hidden,
            ksize=[1, DOC_LENGTH - window_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

    #  全连接层
    squeezed_pool = tf.squeeze(pool, [1, 2])
    raw_output = tf.layers.dense(squeezed_pool, 2)
    output = tf.nn.softmax(raw_output)

    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=raw_output, labels=labels))
    return output, cost


def evaluate_model(sess, output, feed, labels_, print_matrix=False):
    """评估模型指标, 并打印输出"""
    output_values = sess.run(output, feed_dict=feed)
    preds = np.asarray((output_values[:, 1] > 0.5), dtype=int)
    mat = sess.run(tf.confusion_matrix(labels_, preds))
    tn, fp, fn, tp = mat.reshape(4)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if print_matrix:
        print('  confusion matrix:\n', mat)
    print('  precision {:.3f}, recall {:.3f}'.format(precision, recall))


def main():

    # get arguments
    assert len(sys.argv) == 3, __doc__
    train_file, test_file = sys.argv[1:]


    # read data
    print('\n** reading data from files ...\n')
    vocab = build_vocab(train_file)
    inputs_train, labels_train = generate_inputs_and_labels(train_file, vocab)
    inputs_test, labels_test = generate_inputs_and_labels(test_file, vocab)   


    # build convnet
    # 输入词序号 -> 词向量 -> 卷积层(tanh) -> pool 层 -> 全连接层(sigmoid) -> 输出分类
    print('\n** building model ...\n')
    tf.reset_default_graph()
    inputs, labels = generate_placeholders()
    output, cost = build_convnet(vocab, inputs, labels)


    # train
    print('\n** start training (SGD) ...\n')
    feed_train = {inputs: inputs_train, labels: labels_train}
    feed_test = {inputs: inputs_test, labels: labels_test}

    start_time = time.time()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    num_inputs = len(labels_train)
    order = np.arange(num_inputs)
    np.random.shuffle(order)

    try:
        for i in range(rounds):
            if i % print_cost_every == 0:
                cost_train = sess.run(cost, feed_dict=feed_train)
                cost_test = sess.run(cost, feed_dict=feed_test)
                print('round {:03d} cost: train {:.5f} / test {:.5f}'.format(
                    i, cost_train, cost_test))
                evaluate_model(sess, output, feed_test, labels_test)
            for j in range(0, num_inputs, batch_size):
                batch_index = order[j: j + batch_size]
                batch_inputs = inputs_train[batch_index]
                batch_labels = labels_train[batch_index]
                batch_feed = {inputs: batch_inputs, labels: batch_labels}
                sess.run(train_step, feed_dict=batch_feed)
    except KeyboardInterrupt:  # 根据过程输出信息, 若判断已经收敛, 可以 Control-C 中止
        print('Interrupted')
    finally:
        end_time = time.time()
        print('\ntime: {:.2f} s'.format(end_time - start_time))


    # evaluate
    print('\n** evaluating on test set ...\n')
    evaluate_model(sess, output, feed_test, labels_test, print_matrix=True)


if __name__ == '__main__':
    main()
