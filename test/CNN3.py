# coding:utf-8
'''
Created on 2017/9/27.

@author: chk01
'''
import tensorflow as tf
import time
import scipy.io as scio
import random


def get_random_block_from_data(data, batch_size):
    num = len(data['X'])
    if batch_size == -1:
        randomlist = range(num)
    else:
        randomlist = random.sample(range(num), batch_size)
    return [data['X'][i] for i in randomlist], [data['Y'][i] for i in randomlist]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.5)(initial))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def cnn_train(learning_rate, train_epochs):
    sess = tf.InteractiveSession()
    data_train = scio.loadmat('resource/face_train_data.mat')
    data_test = scio.loadmat('resource/face_test_data.mat')
    # global_step = tf.Variable(0)
    # learning_rate = tf.train.exponential_decay(0.001, global_step, 50, 0.98, staircase=True)
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 9], name='y-input')

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('x-image', x_image, 10)

    with tf.name_scope("layer1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("layer2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 9])
    b_fc2 = bias_variable([9])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    tf.summary.histogram("y_conv", y_conv)

    with tf.name_scope("cross_entropy"):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        tf.add_to_collection('loss', cross_entropy)

    loss = tf.add_n(tf.get_collection('loss'))
    tf.summary.scalar('loss', loss)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('resource/summary/style/cnn/train', sess.graph)
    tf.global_variables_initializer().run()

    for step in range(train_epochs):
        xs_batch, ys_batch = get_random_block_from_data(data_train, 50)
        summary = sess.run(merged, feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 1.0})
        train_writer.add_summary(summary, step)
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 1.0})
            print("step %d,training accuracy %g" % (step, train_accuracy))
        train_step.run(feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 0.5})

    xs_batch_test, ys_batch_test = get_random_block_from_data(data_test, -1)
    print("test accuracy %g" % accuracy.eval(feed_dict={x: xs_batch_test, y_: ys_batch_test, keep_prob: 1.0}))
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "resource/model/style/cnn/model.ckpt")
    return True


if __name__ == '__main__':
    cnn_train()
