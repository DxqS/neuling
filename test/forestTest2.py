# coding:utf-8
'''
Created on 2017/9/15.

@author: chk01
'''
import tensorflow as tf
import numpy as  np
from tensorflow.contrib.tensor_forest.client import random_forest

validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key='probabilities'
        ),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key='probabilities'
        ),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key='probabilities'
        )
}
model_dir = 'temp/summary'
hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_trees=3, max_nodes=1000, num_classes=3, num_features=4)
classifier = random_forest.TensorForestEstimator(hparams, model_dir=model_dir,
                                                 config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))

iris = tf.contrib.learn.datasets.load_iris()
data = iris.data.astype(np.float32)
target = iris.target.astype(np.int)


def train_input_fn():
    return data, target



classifier.fit(input_fn=train_input_fn, steps=100)
classifier.evaluate(input_fn=train_input_fn, steps=10, metrics=validation_metrics)
