# coding:utf-8
'''
Created on 2017/9/1.

@author: chk01
'''
import tensorflow as tf
import os
import redis
from pymongo import MongoClient
import numpy as np
import yaml
from tensorflow.examples.tutorials.mnist import input_data

LabelToCode = {
    'TMKA': [1, 0, 0, 0, 0, 0, 0, 0, 0], 'MLSS': [0, 1, 0, 0, 0, 0, 0, 0, 0], 'QCJJ': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'ZRYY': [0, 0, 0, 1, 0, 0, 0, 0, 0], 'GYRM': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'ZXCZ': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'LMMR': [0, 0, 0, 0, 0, 0, 1, 0, 0], 'HLGY': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'XDMD': [0, 0, 0, 0, 0, 0, 0, 0, 1],
}
run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 34])
W = tf.Variable(tf.zeros([34, 9]))
b = tf.Variable(tf.zeros([9]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 9])
cross_entry = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entry)
tf.global_variables_initializer().run()

x_list = mdb.face_train_source.find()
i = 1
for j in x_list:
    if i == 1:
        x4 = np.zeros([1, 34])
        y4 = np.zeros([1, 9])
        i += 1
        x3 = []
        for x2 in j['chin']:
            x3.extend(x2)
        x4[1:] = np.transpose(np.array(x3))
        y4[1:] = LabelToCode[j['label']]
        print(x4.shape)
        print(y4.shape)

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# for i in range(1):
#     xs, ys = mnist.train.next_batch(100)
#     train_step.run({x: xs, y_: ys})
#     print(xs.shape)
#     print(ys.shape)
#
for i in range(10):
    print(i)
    train_step.run({x: x4, y_: y4})
