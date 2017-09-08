# coding:utf-8
'''
Created on 2017/9/7.

@author: chk01
'''
from model_variable import number_softmax, number_cnn, style_softmax

number_softmax_sess = number_softmax.sess
number_softmax_x = number_softmax.x
number_softmax_y = number_softmax.y

style_softmax_sess = style_softmax.sess
style_softmax_x = style_softmax.x
style_softmax_y = style_softmax.y

number_cnn_sess = number_cnn.sess
number_cnn_x = number_cnn.x
number_cnn_y = number_cnn.y_conv
number_cnn_keep_prob = number_cnn.keep_prob

# style_cnn_sess = style_cnn.sess
# style_cnn_x = style_cnn.x
# style_cnn_y = style_cnn.y_conv
# style_cnn_keep_prob = style_cnn.keep_prob
