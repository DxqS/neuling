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


# run_mode = os.environ.get('RUN_ENV', 'local')
# srv = yaml.load(open('srv.yml', 'r'))[run_mode]
# pool = redis.ConnectionPool(**srv['redis'])
# rdb = redis.StrictRedis(connection_pool=pool)
#
# mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
# mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
# mdb = mdb[srv['mongo']['db']]


def load_data_mat(file_name):
    if os.path.exists(file_name):
        return True
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
    scio.savemat(file_name, {'X': x, 'Y': y})
    return True


# train_source = mdb.face_train_source.find_one({"_id": 100001})
# out_line = 'http://dxq.neuling.top' + train_source['result']['chin']

out_line = 'test.jpg'
img = Image.open(out_line)
img2 = img.resize([28, 28])
im3 = img2.convert("L")
# im3.show()
x = np.zeros([1, 784])
tt = np.array(im3)
print(tt.shape)
ss = tt.reshape(1, 784)
print(ss)

print(ss.shape)
# img2.save('out.jpg')
# img.show()
# print(out_line)
