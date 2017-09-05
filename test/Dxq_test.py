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
from PIL import Image

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


def load_data_mat(file_name):
    if os.path.exists(file_name):
        return True
    train_source = mdb.face_train_source.find()
    num = train_source.count()
    x = np.zeros([num, 784])
    y = np.zeros([num, 9])
    for i, source in enumerate(train_source):
        out_line = source['result']['chin']
        img = Image.open('..' + out_line)
        img2 = np.array(img.resize([28, 28]).convert("L")).reshape(1, 784)
        x[i + 1:] = img2
        y[i + 1:] = LabelToCode[source['label']]
    scio.savemat(file_name, {'X': x, 'Y': y})
    return True


if __name__ == "__main__":
    data = scio.loadmat('1.mat')
    print(data['X'].shape)
    print(data['Y'].shape)
