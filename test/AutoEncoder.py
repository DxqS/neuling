# coding:utf-8
'''
Created on 2017/8/10.

@author: Dxq
'''
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 标准均匀分布的Xaiver初始化器，其中fan_in 为输入节点数量，fan_out 为输出节点数量
# 通过tf.random_uniform 创建一个（low,high）↓范围的均匀分布，其方差刚好为2/(n_in+n_out)
def xavier_init(fan_in, fan_out, constant=1):
    temp_var = fan_in + fan_out
    low = -constant * np.sqrt(6.0 / temp_var)
    high = constant * np.sqrt(6.0 / temp_var)
    xavier = tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    return xavier


# 去噪自编码器
class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        '''

        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为softplus
        :param optimizer: 优化器，默认为Adam
        :param scale: 高斯噪声系数，默认为0.1
        :return:
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 训练模型feed数据
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 隐含层
        # 1、加噪后数据：self.x + scale * tf.random_normal((n_input,))
        # 2、wx+b
        # 3、transfer激活
        with tf.name_scope("hidden"):
            self.hidden = self.transfer(
                tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                 self.weights['w1']),
                       self.weights['b1'])
            )
        tf.summary.image('hid', tf.reshape(self.hidden, [-1, 10, 10, 1]), 10)
        # 重构层
        # self.hidden 等于重构层的input输入
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # 定义损失函数：重构层的输出和原始输入x的平方误差
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        tf.summary.scalar("cost", self.cost)
        # 定义优化器进行训练
        self.optimizer = optimizer.minimize(self.cost)

        # 创建session，初始化变量
        self.sess = tf.Session()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('train', self.sess.graph)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        # 提取高阶特征参数
        # w1:xaiver初始化
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))

        # 还原重构参数
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input]), dtype=tf.float32)
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, i):
        '''
        定义计算损失函数cost及执行一步训练
        :param X: 训练模型feed数据
        :return:cost 损失值
        '''
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        summary = self.sess.run(self.merged, feed_dict={self.x: X, self.scale: self.training_scale})
        self.train_writer.add_summary(summary, i)
        return cost, summary

    # 以下暂时未仔细看
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeight(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


mnist = input_data.read_data_sets('MNIST', one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 100
batch_size = 128
display_step = 1
autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input=784, n_hidden=100, transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost, summary = autoencoder.partial_fit(batch_xs, i)
        transform = autoencoder.transform(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print(tf.reshape(transform, [-1, 10, 10, 1]))


# 保存模型
# saver = tf.train.Saver(tf.global_variables())
# saver.save(autoencoder.sess, "model/AutoEncoder.ckpt")
