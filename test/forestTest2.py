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
print('iris', iris)
data = iris.data.astype(np.float32)
target = iris.target.astype(np.int)
print('data', data)

# def train_input_fn():
#     continuous_cols = {
#         k: tf.expand_dims(tf.constant(data.astype(np.float32).values), 1)
#         for k in CONTINUOUS_COLUMNS
#     }
#     # Creates a dictionary mapping from each categorical feature column name (k)
#     # to the values of that column stored in a tf.SparseTensor.
#     categorical_cols = {
#         k: tf.SparseTensor(
#             indices=[[i, 0] for i in range(df[k].size)],
#             values=df[k].values,
#             dense_shape=[df[k].size, 1])
#         for k in CATEGORICAL_COLUMNS
#     }
#     # Merges the two dictionaries into one.
#     feature_cols = dict(continuous_cols.items() + categorical_cols.items())
#     # Add example id list
#     # Converts the label column into a constant Tensor.
#     label = tf.constant(target.values)
#     # Returns the feature columns and the label.
#     return feature_cols, label



classifier.fit(input_fn=train_input_fn, steps=100)
classifier.evaluate(input_fn=train_input_fn, steps=10, metrics=validation_metrics)
