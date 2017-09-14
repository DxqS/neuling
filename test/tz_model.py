# coding:utf-8
'''
Created on 2017/9/13.

@author: chk01
'''
import numpy as np
import os
import yaml
import redis
from pymongo import MongoClient
import scipy.io as scio

SenseToCode = {
    '大': [1, 0, 0], '中': [0, 1, 0], '小': [0, 0, 1]
}

run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]


def load_data_mat(file_name):
    if os.path.exists(file_name):
        return True
    train_source = mdb.style_source.find()
    num = train_source.count()
    x = np.zeros([num, 2])
    y = np.zeros([num, 3])
    for i, source in enumerate(train_source):
        x[i + 1:] = np.array([source['eye_dis_ratio'], source['area'] * 1.0 / 30179.07])
        y[i + 1:] = SenseToCode[source['sense']]
    scio.savemat(file_name, {'X': x, 'Y': y})
    return True


if __name__ == "__main__":
    # load_data_mat('tz_data.mat')
    data = scio.loadmat('tz_data.mat')
    batch_size=1
    import random
    num = len(data['X'])
    randomlist = random.sample(range(55), batch_size)
    print(randomlist)
