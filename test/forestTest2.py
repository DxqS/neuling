# coding:utf-8
'''
Created on 2017/9/15.

@author: chk01
'''
import tensorflow as tf
import numpy as  np
from tensorflow.contrib.tensor_forest.client import random_forest

hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_trees=3, max_nodes=1000, num_classes=3, num_features=4)
classifier = random_forest.TensorForestEstimator(hparams)

iris = tf.contrib.learn.datasets.load_iris()
data = iris.data.astype(np.float32)
target = iris.target.astype(np.float32)

classifier.fit(x=data, y=target, steps=100)
classifier.evaluate(x=data, y=target, steps=10)
