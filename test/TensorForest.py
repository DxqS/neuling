# coding:utf-8
'''
Created on 2017/9/15.

@author: chk01
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib.tensor_forest.client import random_forest

tf.logging.set_verbosity(tf.logging.INFO)

# categorical base columns
# outline = tf.contrib.layers.sparse_column_with_keys(
#     column_name='outline', key=['big', 'mid', 'small'])
# sense = tf.contrib.layers.sparse_column_with_hash_bucket(
#     column_name='sense', key=['big', 'mid', 'small'])

sex = tf.contrib.layers.sparse_column_with_keys(
    column_name='sex', keys=['Female', 'Male'])
detailed_industry_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_industry_recode', hash_bucket_size=1000)

# Continuous base columns
age = tf.contrib.layers.real_valued_column('age')
age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

COLUMNS = ['age', 'detailed_industry_recode', 'sex', 'label']
FEATURE_COLUMNS = [age, age_buckets, detailed_industry_recode, sex]
# FEATURE_COLUMNS = [age, detailed_occupation_recode, education, wage_per_hour]

LABEL_COLUMN = 'label'
CONTINUOUS_COLUMNS = [
    'age'
]
CATEGORICAL_COLUMNS = [
    'detailed_industry_recode', 'sex'
]

TRAIN_FILE = 'census-income.data'
TEST_FILE = 'census-income.test'

df_train = pd.read_csv(TRAIN_FILE, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(TEST_FILE, names=COLUMNS, skipinitialspace=True)
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)
df_train[['detailed_industry_recode']] = df_train[['detailed_industry_recode']].astype(str)
df_test[['detailed_occupation_recode']] = df_test[['detailed_industry_recode']].astype(str)

df_train[LABEL_COLUMN] = (
    df_train[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
df_test[LABEL_COLUMN] = (
    df_test[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
# print df_train.dtypes
dtypess = df_train.dtypes

# print dtypess[CATEGORICAL_COLUMNS]

print(df_train.head(5))
print(df_test.head(5))


def input_fn(df):
    continuous_cols = {
        k: tf.expand_dims(tf.constant(df[k].astype(np.float32).values), 1)
        for k in CONTINUOUS_COLUMNS
    }
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS
    }
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Add example id list
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)


model_dir = '../rf_model_dir'
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

hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_trees=10,
    max_nodes=1000,
    num_classes=2,
    num_features=len(CONTINUOUS_COLUMNS) + len(CATEGORICAL_COLUMNS))
classifier = random_forest.TensorForestEstimator(hparams, model_dir=model_dir,
                                                 config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))

classifier.fit(input_fn=train_input_fn, steps=200)
results = classifier.evaluate(
    input_fn=eval_input_fn, steps=1, metrics=validation_metrics)
print(results)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
