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
from PIL import Image, ImageDraw
import pandas as pd

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


face_struct = [
    (59, 499),
    (64, 555),
    (70, 616),
    (83, 677),
    (108, 734),
    (135, 777),
    (171, 825),
    (221, 871),
    (301, 896),
    (365, 879),
    (416, 839),
    (456, 793),
    (491, 739),
    (513, 687),
    (531, 616),
    (536, 558),
    (539, 486)
]
face_struct_eye = [
    (189, 500),
    (412, 500)
]
en_ch = {
    'big': "大",  # num:754 avg:0.44122
    'mid': "中",  # num:456 avg:0.43868
    'small': "小",  # num:161 avg:0.44235
}


def draw_points(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    print(x)
    print(y)
    min_x = min(x) - 5
    min_y = min(y) - 5
    max_x = max(x) + 5
    max_y = max(y) + 5
    wid = max(max_y - min_y, max_x - min_x)

    pil_image = Image.open('../static/images/1.jpg')
    d = ImageDraw.Draw(pil_image)
    d.line(points, width=5)
    d.line(face_struct_eye, width=5)
    region = pil_image.crop((min_x, min_y, min_x + wid, min_y + wid))
    res = region.resize((300, 300), Image.ANTIALIAS).convert("L")
    res.save('res.jpg')
    return True


def read_data_pandas():
    import random
    SenseToNum = {'小': 0, '中': 1, '大': 2}
    sources = mdb.style_source.find({"type": "train"})
    train_data_ratio = []
    train_data_area = []
    train_data_label = []
    test_data_ratio = []
    test_data_area = []
    test_data_label = []
    randomlist = random.sample(range(sources.count()), 1000)
    for i, source in enumerate(sources):
        if i in randomlist:
            train_data_ratio.append(source['eye_dis_ratio'])
            train_data_area.append(source['area'])
            train_data_label.append(SenseToNum[source['sense']])
        else:
            test_data_ratio.append(source['eye_dis_ratio'])
            test_data_area.append(source['area'])
            test_data_label.append(SenseToNum[source['sense']])

    train_save = pd.DataFrame({'ratio': train_data_ratio, 'area': train_data_area, 'label': train_data_label},
                              index=randomlist)
    train_save.index.name = 'index'
    test_save = pd.DataFrame({'ratio': test_data_ratio, 'area': test_data_area, 'label': test_data_label})

    train_save.sort_index().to_csv('style.train', index=True)
    test_save.to_csv('style.test', index=True)
    return True


if __name__ == "__main__":
    read_data_pandas()
