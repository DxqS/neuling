# coding:utf-8
'''
Created on 2017/9/20.

@author: chk01
'''
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
y = tf.constant(value=[[.5, .2], [.2, .3], [.3, .5]], shape=[3, 2], dtype=tf.float32)
y_ = np.array([[.3, .5], [.6, .2], [.1, .3]])
print(y_.shape)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
cross_entropy2 = -tf.reduce_sum(y_ * tf.log(y), axis=1)

print(sess.run(cross_entropy))
print(sess.run(tf.reduce_mean(cross_entropy)))

print(sess.run(cross_entropy2))
print(sess.run(tf.reduce_mean(cross_entropy2)))
