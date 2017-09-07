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
saver.restore(sess, "resource/model/number/softmax/model.ckpt")
