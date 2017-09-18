# coding:utf-8
'''
Created on 2017/9/16.

@author: chk01
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import shutil
from tensorflow.contrib.tensor_forest.client import random_forest

# from tensorflow.contrib.learn.python.learn.estimators import random_forest as random_forest2

ratio = tf.contrib.layers.real_valued_column('ratio')
ratio_buckets = tf.contrib.layers.bucketized_column(
    ratio, boundaries=[0.445, 0.460, 0.5])

area = tf.contrib.layers.real_valued_column('area')
area_buckets = tf.contrib.layers.bucketized_column(
    area, boundaries=[25000.0, 27000.2, 32500.0])

COLUMNS = [
    'ratio', 'area', 'label'
]

# FEATURE_COLUMNS = [age, detailed_occupation_recode, education, wage_per_hour]
#
CATEGORICAL_COLUMNS = [
    'ratio', 'area'
]
TRAIN_FILE = 'style.train'
TEST_FILE = 'style.test'

# pd 读取csv,按照names 排序
# skipinitialspace=True分隔符后跳过空格
df_train = pd.read_csv(TRAIN_FILE, skipinitialspace=True)
df_test = pd.read_csv(TEST_FILE, skipinitialspace=True)
# 数据预处理，去除行中有一个为NAN的数据
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)


def input_fn(df):
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    continuous_cols = {
        k: tf.expand_dims(tf.constant(df[k].astype(np.float32).values), 1)
        for k in CATEGORICAL_COLUMNS
    }
    # Merges the two dictionaries into one.
    feature_cols = continuous_cols
    # Add example id list
    # Converts the label column into a constant Tensor.
    label = tf.constant(df['label'].values)
    # Returns the feature columns and the label.
    return feature_cols, label


#
#
def train_input_fn():
    return input_fn(df_train)


#
#
def eval_input_fn():
    return input_fn(df_test)


#
#
model_dir = 'train'
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key='probabilities'
        ),
    # "precision":
    #     tf.contrib.learn.MetricSpec(
    #         metric_fn=tf.contrib.metrics.streaming_precision,
    #         prediction_key='probabilities'
    #     ),
    # "recall":
    #     tf.contrib.learn.MetricSpec(
    #         metric_fn=tf.contrib.metrics.streaming_recall,
    #         prediction_key='probabilities'
    #     )
}
#
hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_trees=6,
    max_nodes=50,
    num_classes=3,
    num_features=2)
early_stopping_rounds = 100
check_every_n_steps = 100
# monitor = random_forest.TensorForestLossHook(early_stopping_rounds, check_every_n_steps)
# num_trees=100,
# max_nodes=10000,
# bagging_fraction=1.0,
# num_splits_to_consider=0,
# feature_bagging_fraction=1.0,
# max_fertile_nodes=0,
# split_after_samples=250,
# min_split_samples=5,
# valid_leaf_threshold=1,
# dominate_method='bootstrap',
# dominate_fraction=0.99,
config = {
    'master': None,  # 运行分支默认为local
    'num_cores': 0,
    'log_device_placement': False,
    'gpu_memory_fraction': 1,  # 0-1GPU使用占比
    'tf_random_seed': None,
    'save_summary_steps': 100,
    'save_checkpoints_secs': 0,
    'save_checkpoints_steps': None,
    'keep_checkpoint_max': 1,  # 允许保存的checkpoint数量
    'keep_checkpoint_every_n_hours': 10000,
    'log_step_count_steps': 100,
    'evaluation_master': '',
    'model_dir': model_dir,
    'session_config': None
}
classifier = random_forest.TensorForestEstimator(hparams, model_dir=model_dir,
                                                 config=tf.contrib.learn.RunConfig(**config))
classifier.fit(input_fn=train_input_fn, steps=200)
results = classifier.evaluate(
    input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


results2 = classifier.evaluate(
    input_fn=train_input_fn, steps=2)
for key in sorted(results2):
    print("%s: %s" % (key, results2[key]))

# 训练和评估模型
# print(train_input_fn())
# model = tf.contrib.learn.LinearClassifier(
#     feature_columns=FEATURE_COLUMNS, model_dir=model_dir)
# model.fit(input_fn=train_input_fn, steps=200)
# results = model.evaluate(input_fn=eval_input_fn, steps=1)
# for key in sorted(results):
#     print("%s: %s" % (key, results[key]))
