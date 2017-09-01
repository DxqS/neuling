# coding:utf-8
'''
Created on 2017/9/1.

@author: chk01
'''
import tensorflow as tf
import config

mdb = config.mdb

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 400 * 400])
W = tf.Variable(tf.zeros(400 * 400, 9))
b = tf.Variable(tf.zeros([9]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 9])
cross_entry = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entry)
tf.global_variables_initializer().run()

x_list = mdb.face_train_source.find()
for x in x_list:
    print(x['_id'])

for i in range(10):
    train_step.run({x: 1, y_: 1})
