# coding:utf-8
'''
Created on 2017/8/5.

@author: Dxq
'''
import tensorflow as tf
import numpy as np

# 读取数据
# f = open("min2_data.txt", "r")
# lines = f.readlines()  # 读取全部内容
# for i, line in enumerate(lines):
#     if i == 0:
#         x_input = np.transpose(np.array(list(map(int, line.split(',')))))
#     else:
#         y_input = np.transpose(np.array(list(map(int, line.split(',')))))
g = tf.Graph()
with g.as_default():
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 1], name='x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y')
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W', trainable=True)
    b = tf.Variable(tf.zeros([1]), name="b")

    y = tf.add(tf.multiply(x, W), 0, name='last')

    cross_entropy_mean = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy_mean)
    tf.global_variables_initializer().run()

    x_data = np.random.rand(10, 1).astype(np.float32)
    y_data = x_data + .3
    for step in range(3000):
        train_step.run({x: x_data, y_: y_data})
        if step % 20 == 0:
            print(step, sess.run(W)[0])
    graph_def = g.as_graph_def()
    tf.train.write_graph(graph_def, './', 'expert-graph3.pb', as_text=False)
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "model/model.ckpt")




#
# x = tf.placeholder(tf.float32, [None, 1])
# W = tf.Variable(tf.zeros([1]))
# b = tf.Variable(tf.zeros([1]))
# y = tf.nn.softmax(tf.multiply(x, W) + b)
# y_ = tf.placeholder(tf.float32, [None, 1])
# # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
# cross_entropy_mean = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_mean)
# tf.global_variables_initializer().run()
#
# for i in range(10):
#     train_step.run({x: x_input, y_: y_input})
#     print(sess.run(W))
#     print(sess.run(b))
