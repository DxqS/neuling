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
import random

# from tensorflow.examples.tutorials.mnist import input_data

LabelToCode = {
    'TMKA': [5, 2, 0,
             2, 1, 0,
             0, 0, 0],
    'MLSS': [.15, .55, .15,
             0, .15, 0,
             0, 0, 0],
    'QCJJ': [0, .2, .5,
             0, .1, .2,
             0, 0, 0],
    'ZRYY': [.15, 0, 0,
             .55, .15, 0,
             .15, 0, 0],
    'GYRM': [0, .1, 0,
             .1, .6, .1,
             0, .1, 0],
    'ZXCZ': [0, 0, .15,
             0, .15, .55,
             0, 0, .15],
    'LMMR': [0, 0, 0,
             .2, .1, 0,
             .5, .2, 0],
    'HLGY': [0, 0, 0,
             0, .15, 0,
             .15, .55, .15],
    'XDMD': [0, 0, 0,
             0, .1, .2,
             0, .2, .5],
}

run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]

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


def dis(p1, p2):
    import math
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


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
    SenseToNum = {'小': 0, '中': int(np.random.random(1) > 0.5), '大': 1}
    StyleToNum = {'TMKA': 0, 'MLSS': 1, 'QCJJ': 2,
                  'ZRYY': 3, 'GYRM': 4, 'ZXCZ': 5,
                  'LMMR': 6, 'HLGY': 7, 'XDMD': 8}
    sources = mdb.style_source.find({"type": "train"})
    train_data_ratio = []
    train_data_area = []
    train_data_type = []
    train_data_label = []

    train_left_eye_ratio = []
    train_right_eye_ratio = []
    train_nose_ratio = []
    train_lip_ratio = []
    train_line_1_left_ratio = []
    train_line_1_right_ratio = []
    train_line_2_ratio = []

    test_data_type = []
    test_data_ratio = []
    test_data_area = []
    test_data_label = []

    test_left_eye_ratio = []
    test_right_eye_ratio = []
    test_nose_ratio = []
    test_lip_ratio = []
    test_line_1_left_ratio = []
    test_line_1_right_ratio = []
    test_line_2_ratio = []

    randomlist = random.sample(range(sources.count()), 1000)
    for i, source in enumerate(sources):
        if i in randomlist:
            train_data_type.append(StyleToNum[source['label']])
            train_data_ratio.append(source['eye_dis_ratio'])
            train_data_area.append(source['area'])
            train_data_label.append(SenseToNum[source['sense']])

            train_left_eye_ratio.append(source['left_eye_ratio'])
            train_right_eye_ratio.append(source['right_eye_ratio'])
            train_nose_ratio.append(source['nose_ratio'])
            train_lip_ratio.append(source['lip_ratio'])
            train_line_1_left_ratio.append(source['line_1_left_ratio'])
            train_line_1_right_ratio.append(source['line_1_right_ratio'])
            train_line_2_ratio.append(source['line_2_ratio'])

        else:
            test_data_type.append(StyleToNum[source['label']])
            test_data_ratio.append(source['eye_dis_ratio'])
            test_data_area.append(source['area'])
            test_data_label.append(SenseToNum[source['sense']])

            test_left_eye_ratio.append(source['left_eye_ratio'])
            test_right_eye_ratio.append(source['right_eye_ratio'])
            test_nose_ratio.append(source['nose_ratio'])
            test_lip_ratio.append(source['lip_ratio'])
            test_line_1_left_ratio.append(source['line_1_left_ratio'])
            test_line_1_right_ratio.append(source['line_1_right_ratio'])
            test_line_2_ratio.append(source['line_2_ratio'])

    train_save = pd.DataFrame(
        {'area': train_data_area, 'ratio': train_data_ratio, 'style': train_data_type, 'label': train_data_label,
         'left_eye_ratio': train_left_eye_ratio, 'right_eye_ratio': train_right_eye_ratio,
         'nose_ratio': train_nose_ratio,
         'lip_ratio': train_lip_ratio, 'line_1_left_ratio': train_line_1_left_ratio,
         'line_1_right_ratio': train_line_1_right_ratio, 'line_2_ratio': train_line_2_ratio
         },
        index=randomlist)
    train_save.index.name = 'index'
    test_save = pd.DataFrame(
        {'area': test_data_area, 'ratio': test_data_ratio, 'style': test_data_type, 'label': test_data_label,
         'left_eye_ratio': test_left_eye_ratio, 'right_eye_ratio': test_right_eye_ratio, 'nose_ratio': test_nose_ratio,
         'lip_ratio': test_lip_ratio, 'line_1_left_ratio': test_line_1_left_ratio,
         'line_1_right_ratio': test_line_1_right_ratio, 'line_2_ratio': test_line_2_ratio
         })

    train_save.sort_index().to_csv('style.train', index=True)
    test_save.to_csv('style.test', index=True)
    return True


def get_k_chin(p0, p1, p2):
    # 计算过三点的抛物线，输出在顶点处的曲率
    w = np.array([[np.power(p0[0], 2), p0[0], 1],
                  [np.power(p1[0], 2), p1[0], 1],
                  [np.power(p2[0], 2), p2[0], 1]])
    res = np.matrix(w).I * np.array([[p0[1]], [p1[1]], [p2[1]]])
    a, b, c = float(np.array(res)[0][0]), float(np.array(res)[1][0]), float(np.array(res)[2][0])
    return np.abs(float(2 * a))


def get_rec_ratio(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    return round((max_y - min_y) / (max_x - min_x), 3)


def get_face_data():
    sources = mdb.style_source.find()
    for source in sources:
        nose_tip = source['nose_tip']
        left_eye = source['left_eye']
        left_eyebrow = source['left_eyebrow']
        top_lip = source['top_lip']
        nose_bridge = source['nose_bridge']
        right_eyebrow = source['right_eyebrow']
        bottom_lip = source['bottom_lip']
        right_eye = source['right_eye']
        chin = source['chin']

        left_eye_ratio = get_rec_ratio(left_eye)
        right_eye_ratio = get_rec_ratio(right_eye)
        nose = nose_tip
        nose.extend(nose_bridge)
        nose_ratio = get_rec_ratio(nose)

        lip = top_lip
        lip.extend(bottom_lip)
        lip_ratio = get_rec_ratio(lip)

        line_1_left = [chin[3], nose_tip[0], chin[7], left_eye[-2]]
        line_1_left_ratio = round(get_rec_ratio(line_1_left), 3)
        line_1_right = [chin[-4], nose_tip[-1], chin[9], right_eye[-2]]
        line_1_right_ratio = round(get_rec_ratio(line_1_right), 3)

        line_2 = [chin[0], chin[-1], chin[8]]
        line_2.extend(left_eyebrow)
        line_2.extend(right_eyebrow)
        line_2_ratio = get_rec_ratio(line_2)

        mdb.style_source.update_one({"_id": source['_id']}, {"$set": {
            'left_eye_ratio': left_eye_ratio,
            'right_eye_ratio': right_eye_ratio,
            'nose_ratio': nose_ratio,
            'lip_ratio': lip_ratio,
            'line_1_left_ratio': line_1_left_ratio,
            'line_1_right_ratio': line_1_right_ratio,
            'line_2_ratio': line_2_ratio
        }})
    return True


def load_data_mat(file_name):
    if os.path.exists(file_name):
        return True
    train_source = mdb.style_source.find()
    num = train_source.count()
    x = np.zeros([num, 784])
    y = np.zeros([num, 9])
    for i, source in enumerate(train_source):
        out_line = source['result']['chin']
        img = Image.open('..' + out_line)
        img2 = np.array(img.resize([28, 28]).convert("L")).reshape(1, 784).astype(np.float32)
        image = np.multiply(img2, 1.0 / 255.0)
        x[i + 1:] = image
        y[i + 1:] = LabelToCode[source['label']]
    scio.savemat(file_name, {'X': x, 'Y': y})
    return True


def style_data():
    import random
    sources = mdb.style_source.find()
    num = sources.count()
    randomlist = random.sample(range(num), num - 1000)
    data = scio.loadmat('face_data2.mat')
    train_x = np.zeros([1000, 784])
    train_y = np.zeros([1000, 9])
    test_x = np.zeros([num - 1000, 784])
    test_y = np.zeros([num - 1000, 9])
    test = 0
    train = 0
    for i, source in enumerate(sources):

        if i in randomlist:
            try:
                test += 1
                test_x[test:] = data['X'][i]
                test_y[test:] = data['Y'][i]
            except:
                pass
        else:
            try:
                train += 1
                train_x[train:] = data['X'][i]
                train_y[train:] = data['Y'][i]
            except:
                pass
    scio.savemat('face_data_test.mat', {'X': test_x, 'Y': test_y})
    scio.savemat('face_data_train.mat', {'X': train_x, 'Y': train_y})

    return True


def load_outline_data(file_name):
    if os.path.exists(file_name):
        return True
    QualityToNum = {"曲": [1, 0, 0], "中": [0, 1, 0], "直": [0, 0, 1]}
    sources = mdb.style_source.find()
    num = sources.count()
    randomlist = random.sample(range(num), num - 1000)

    train_x = np.zeros([1000, 784])
    train_y = np.zeros([1000, 3])
    test_x = np.zeros([num - 1000, 784])
    test_y = np.zeros([num - 1000, 3])
    test = 0
    train = 0
    for i, source in enumerate(sources):
        out_line = source['result']['chin']
        img = Image.open('..' + out_line)
        img2 = np.array(img.resize([28, 28]).convert("L")).reshape(1, 784).astype(np.float32)
        image = np.multiply(img2, 1.0 / 255.0)
        if i in randomlist:
            try:
                test_x[test:] = image
                test_y[test:] = QualityToNum[source['outline']]
                test += 1
            except:
                print(test)
        else:
            try:

                train_x[train:] = image
                train_y[train:] = QualityToNum[source['outline']]
                train += 1
            except:
                print(train)
    scio.savemat(file_name.replace('train', 'test'), {'X': test_x, 'Y': test_y})
    scio.savemat(file_name, {'X': train_x, 'Y': train_y})

    return True


if __name__ == "__main__":
    load_outline_data("style_outline.train")

    # df_train = pd.read_csv("style_outline.train", skipinitialspace=True)
    # print(np.array(df_train['outline'])[1].shape)
