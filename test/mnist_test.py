# coding:utf-8
'''
Created on 2017/9/6.

@author: chk01
'''
import tensorflow as tf
import numpy as np
from PIL import Image


def test(x_input):
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    saver = tf.train.Saver()
    saver.restore(sess, "model/face.ckpt")
    res = sess.run(y, feed_dict={x: x_input})
    return res


if __name__ == "__main__":
    img = np.array(Image.open('5.png').convert("L").resize((28, 28), Image.ANTIALIAS)).reshape(1, 784)
    print(test(img))
