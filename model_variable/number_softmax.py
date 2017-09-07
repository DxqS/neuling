# coding:utf-8
'''
Created on 2017/9/7.

@author: chk01
'''
import tensorflow as tf

sess = tf.InteractiveSession()

# Softmax
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()
saver.restore(sess, "resource/model/number.ckpt")




# sess = tf.InteractiveSession()

# Softmax
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# CNN
# x = tf.placeholder(tf.float32, [None, 784])
# y_ = tf.placeholder(tf.float32, [None, 10])
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# W_conv1 = tf_service.weight_variable([5, 5, 1, 32])
# b_conv1 = tf_service.bias_variable([32])
# h_conv1 = tf.nn.relu(tf_service.conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = tf_service.max_pool_2x2(h_conv1)
#
# W_conv2 = tf_service.weight_variable([5, 5, 32, 64])
# b_conv2 = tf_service.bias_variable([64])
# h_conv2 = tf.nn.relu(tf_service.conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = tf_service.max_pool_2x2(h_conv2)
#
# W_fc1 = tf_service.weight_variable([7 * 7 * 64, 1024])
# b_fc1 = tf_service.bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = tf_service.weight_variable([1024, 10])
# b_fc2 = tf_service.bias_variable([10])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# saver = tf.train.Saver()
# saver.restore(sess, "resource/model/number.ckpt")
