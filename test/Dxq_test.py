# coding:utf-8
'''
Created on 2017/6/26.

@author: Dxq
'''
import numpy as np
import os
import yaml
import redis
from pymongo import MongoClient
import scipy.io as scio

LabelToCode = {
    'TMKA': [1, 0, 0, 0, 0, 0, 0, 0, 0], 'MLSS': [0, 1, 0, 0, 0, 0, 0, 0, 0], 'QCJJ': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'ZRYY': [0, 0, 0, 1, 0, 0, 0, 0, 0], 'GYRM': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'ZXCZ': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'LMMR': [0, 0, 0, 0, 0, 0, 1, 0, 0], 'HLGY': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'XDMD': [0, 0, 0, 0, 0, 0, 0, 0, 1],
}
run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]


def get_random_block_from_data(data, batch_size):
    num = len(data['X'])
    start_index = np.random.randint(0, num - batch_size)
    return data['X'][start_index:(start_index + batch_size)], data['Y'][start_index:(start_index + batch_size)]


def load_data_mat():
    train_source = mdb.face_train_source.find()
    num = train_source.count()
    x = np.zeros([num, 34])
    y = np.zeros([num, 9])
    for i, source in enumerate(train_source):
        xs = []
        for point in source['chin']:
            xs.extend(point)
        x[i + 1:] = np.transpose(np.array(xs))
        y[i + 1:] = LabelToCode[source['label']]
    dataNew = 'data.mat'
    scio.savemat(dataNew, {'X': x, 'Y': y})


dataNew = 'data.mat'
data = scio.loadmat(dataNew)
# print(len(data['X']))
xs_bath, ys_bath = get_random_block_from_data(data, 10)
print(xs_bath)
print(ys_bath)