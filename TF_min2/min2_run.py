# coding:utf-8
'''
Created on 2017/8/5.

@author: Dxq
'''
import os
import sys
import argparse
import tensorflow as tf

current_dir_path = os.path.dirname(os.path.realpath(__file__))


def run(x_input):
    # 定义变量
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 1], name='x')
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W', trainable=True)
    b = tf.Variable(tf.zeros([1]), name="b")

    # 结果
    y = tf.add(tf.multiply(x, W), b, name='last')

    saver = tf.train.Saver()
    saver.restore(sess, "model/min2.ckpt")
    res = sess.run(y, feed_dict={x: x_input})[0][0]
    return res


def main(_):
    print('res', run([[FLAGS.x]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--x',
        type=float,
        default=1.0,
        help="""\Default x.\
      """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
