# coding:utf-8
'''
Created on 2017/9/15.

@author: chk01
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensor_forest.client import random_forest

model_dir = 'temp/summary'

params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_trees=3, max_nodes=1000, num_classes=3, num_features=2)

classifier = random_forest.TensorForestEstimator(params, model_dir=model_dir)

# load data
iris = tf.contrib.learn.datasets.load_iris()
data = iris.data.astype(np.float32)
# numpy.ndarray (150,4)

target = iris.target.astype(np.int)


# numpy.ndarray (150,)


def train_input_fn():
    feature_cols = tf.SparseTensor(indices, values, dense_shape)

    # Add example id list
    # Converts the label column into a constant Tensor.
    label = tf.constant(target)
    # Returns the feature columns and the label.
    return feature_cols, label


# ss, tt = train_input_fn()
classifier.fit(input_fn=train_input_fn, steps=100)
# classifier.evaluate(x=data, y=target, steps=10)
