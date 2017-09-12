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

# from tensorflow.examples.tutorials.mnist import input_data

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


def dis(p1, p2):
    import math
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def eye_dis_ratio():
    sources = mdb.style_source.find()
    for i, source in enumerate(sources):
        left_eye = source['left_eye']
        right_eye = source['right_eye']

        mid_len = dis(left_eye[3], right_eye[0])
        est_len = dis(left_eye[0], right_eye[3])
        mdb.style_source.update_one({"_id": source['_id']}, {"$set": {"eye_dis_ratio": round(mid_len / est_len, 5)}})
        if i % 100 == 0:
            print(i)


def find_point():
    big_sense_sample = mdb.style_source.count({"sense": "大"})
    # mid_sense = mdb.style_source.count({"sense": "中"})
    # small_sense = mdb.style_source.count({"sense": "小"})
    for i in range(3789, 5001):
        ratio = i * 1.0 / 10000.0
        big_sense = mdb.style_source.count({"sense": "大", "eye_dis_ratio": {"$lt": ratio}})
        print('accuracy', big_sense * 1.0 / big_sense_sample)
        # mid_sense = mdb.style_source.count({"sense": "中"})
        # small_sense = mdb.style_source.count({"sense": "小"})


if __name__ == "__main__":
    # read_source_to_db("GYRM")
    find_point()
    # load_data_mat('face_data.mat')
    # data = scio.loadmat('face_data.mat')
    # print(data['X'].shape)
    # print(data['Y'].shape)
