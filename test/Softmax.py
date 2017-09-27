# coding:utf-8
'''
Created on 2017/9/20.

@author: chk01
'''
import tensorflow as tf
import random
import scipy.io as scio
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    # tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.5)(initial))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_random_block_from_data(data, batch_size):
    num = len(data['X'])
    if batch_size == -1:
        randomlist = range(num)
    else:
        randomlist = random.sample(range(num), batch_size)
    return [data['X'][i] for i in randomlist], [data['Y'][i] for i in randomlist]


def train():
    data_train = scio.loadmat('face_data2.mat')
    data_test = scio.loadmat('face_data2.mat')
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 9])

    W = weight_variable([784, 9])
    b = bias_variable([9])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    yy = tf.nn.relu(tf.matmul(x, W) + b)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log((1 - y)), axis=1))
    # tf.nn.softmax_cross_entropy_with_logits()
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 开始训练
    xs_batch, ys_batch = get_random_block_from_data(data_train, 50)
    for step in range(1000):
        train_step.run(feed_dict={x: xs_batch, y_: ys_batch})
        if step % 1000 == 0:
            xs_batch_test, ys_batch_test = get_random_block_from_data(data_test, 1)
            print(y.eval({x: xs_batch_test, y_: ys_batch_test})[0])
            print(yy.eval({x: xs_batch_test, y_: ys_batch_test}))

    # # 保存模型
    # saver = tf.train.Saver(tf.global_variables())
    # saver.save(sess, "model/softmax.ckpt")

    return True


if __name__ == "__main__":
    train()
