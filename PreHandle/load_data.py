# coding:utf-8
'''
Created on 2017/8/10.

@author: Dxq
'''
import os
# import cv2
# import dlib
# import config
from PIL import Image, ImageDraw
import face_recognition
import time
# from common import base
import redis
from pymongo import MongoClient

import yaml

run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]

# mdb = config.mdb
# rdb = config.rdb
# PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
current_dir_path = os.path.dirname(os.path.realpath(__file__))
FEATURES = [
    'chin',
    'left_eyebrow',
    'right_eyebrow',
    # 'nose_bridge',
    # 'nose_tip',
    'left_eye',
    'right_eye',
    'top_lip',
    'bottom_lip'
]


def getRedisID(key):
    if rdb.hget('RedisID', key) is None:
        colList = list(mdb[key].find().sort('_id', -1).limit(1))
        ids = 100000 if len(colList) == 0 else colList[0]['_id'] + 1
        rdb.hset('RedisID', key, ids)
    return int(rdb.hincrby('RedisID', key, 1))

def face_landmarks(face_image):
    image = face_recognition.load_image_file(face_image)
    face_landmarks_list = face_recognition.face_landmarks(image)

    return face_landmarks_list[0]


def getLabel(path):
    return path[path.rfind('\\') - 4:path.rfind('\\')]


def drawPoints(points, file_name):
    file_dir = file_name[:file_name.rfind('\\')]
    print (file_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    min_x = min(x) - 5
    min_y = min(y) - 5
    max_x = max(x) + 5
    max_y = max(y) + 5
    wid = max(max_y - min_y, max_x - min_x)

    pil_image = Image.open('1.jpg')
    d = ImageDraw.Draw(pil_image)
    d.line(points, width=5)
    region = pil_image.crop((min_x, min_y, min_x + wid, min_y + wid))
    res = region.resize((300, 300), Image.ANTIALIAS)
    res.save(file_name)
    return True


# if __name__ == '__main__':
source_dir = current_dir_path + '/Source'
for root, dirs, files in os.walk(source_dir):
    i = 0
    for file in files:
        ts = time.time()
        i += 1
        path = os.path.join(root, file)
        if not mdb.face.find_one({"path": path}):
            label = getLabel(path)
            face_landmarks_dict = face_landmarks(path)
            # 暂时只画轮廓
            result = {}
            for feature in FEATURES:
                outline = face_landmarks_dict[feature]
                file_name = path.replace('Source', 'Result\\' + feature)
                print (file_name)
                drawPoints(points=outline, file_name=file_name)
                result[feature] = file_name
            if face_landmarks_dict != "Error":
                face = {
                    '_id': getRedisID("face_train_source"),
                    'path': path,
                    'label': label,
                    'type': 'train',
                    'result': result
                }
                face.update(face_landmarks_dict)
                mdb.face.insert(face)
        print(str(i) + '==耗时==' + str(time.time() - ts))
