# coding:utf-8
'''
Created on 2017/8/4.

@author: Dxq
'''
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

# 定义变量
x = tf.placeholder(tf.float32, [None, 1], name='x')
y_ = tf.placeholder(tf.float32, [None, 1], name='y')
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W', trainable=True)
b = tf.Variable(tf.zeros([1]), name="b")

y = tf.add(tf.multiply(x, W), b, name='last')

# 目标函数
cross_entropy_mean = tf.reduce_mean(tf.square(y - y_))

# 训练算法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy_mean)

tf.global_variables_initializer().run()

# 训练数据集
x_data = np.random.rand(10, 1).astype(np.float32)
y_data = 2 * x_data + .3

# 开始训练
for step in range(1000):
    train_step.run(feed_dict={x: x_data, y_: y_data})
    if step % 20 == 0:
        print(step, sess.run(W)[0])

# 保存模型
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, "model/min2.ckpt")
