# coding:utf-8
'''
Created on 2017/8/23.

@author: Dxq
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

max_steps = 100
learning_rate = 0.01
dropout = 0.9

log_dir = 'model'
mnist = input_data.read_data_sets("MNIST", one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 10)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# def variables_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope("stddev"):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.scalar('histogram', var)
#
#
# def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
#     with tf.name_scope(layer_name):
#         with tf.name_scope("weights"):
#             weights = weight_variable([input_dim, output_dim])
#             variables_summaries(weights)
#         with tf.name_scope("biases"):
#             biases = weight_variable([output_dim])
#             variables_summaries(biases)
#         with tf.name_scope("Wx_plus_b"):
#             preactivate = tf.matmul(input_tensor, weights) + biases
#             tf.summary.histogram('pre_activation', preactivate)
#         activations = act(preactivate, name='activations')
#         tf.summary.histogram('activations', activations)
#         return activations


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = weight_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope("dropout"):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout', keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train_step"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
# test_writer = tf.summary.FileWriter(log_dir + '/test')

tf.global_variables_initializer().run()

saver = tf.train.Saver(tf.global_variables())
for i in range(1):
    batch = mnist.train.next_batch(50)
    print('X_data:')
    print(batch[0])
    print(batch[0].shape)
    print('Y_data:')
    print(batch[1].shape)
    print(batch[1])
    # if i % 100 == 0:
    #     train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    #     print("step %d,training accuracy %g" % (i, train_accuracy))
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # summary = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # train_writer.add_summary(summary, i)
    # if i % 50 == 0:
    #     saver.save(sess, log_dir + "/cnn.ckpt", i)

# print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
train_writer.close()
