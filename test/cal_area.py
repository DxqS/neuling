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
import math

run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def GetAreaOfPolyGon(points):
    '''计算多边形面积值
       points:多边形的点集，每个点为Point类型
       返回：多边形面积'''
    area = 0
    if len(points) < 3:
        raise Exception("至少需要3个点才有面积")

    for i in range(2):
        if i == 0:
            p1 = points[0]
            p2 = points[1]
            p3 = points[2]
        else:
            p1 = points[2]
            p2 = points[3]
            p3 = points[0]
        # 计算向量
        vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
        vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)
        # 判断顺时针还是逆时针，顺时针面积为正，逆时针面积为负
        vecMult = vecp1p2.x * vecp2p3.y - vecp1p2.y * vecp2p3.x

        sign = 0
        if (vecMult > 0):
            sign = 1
        elif (vecMult < 0):
            sign = -1

        triArea = GetAreaOfTriangle(p1, p2, p3) * sign
        area += triArea

    return abs(area)


# 计算三角形面积---海伦公式
def GetAreaOfTriangle(p1, p2, p3):
    p1p2 = GetLineLength(p1, p2)
    p2p3 = GetLineLength(p2, p3)
    p3p1 = GetLineLength(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)  # 海伦公式
    area = math.sqrt(area)
    return area


# 计算边长
def GetLineLength(p1, p2):
    length = math.pow((p1.x - p2.x), 2) + math.pow((p1.y - p2.y), 2)  # pow  次方
    length = math.sqrt(length)
    return length


# face_struct = [
#     (59, 499),
#     (64, 555),
#     (70, 616),
#     (83, 677),
#     (108, 734),
#     (135, 777),
#     (171, 825),
#     (221, 871),
#     (301, 896),
#     (365, 879),
#     (416, 839),
#     (456, 793),
#     (491, 739),
#     (513, 687),
#     (531, 616),
#     (536, 558),
#     (539, 486)
# ]
# face_struct_eye = [
#     (189, 500),
#     (412, 500)
# ]
# S= 30179.07 标准面积

# 大量感 比标准大的722 小的32
# 小量感 比标准大的120 小的41
# 中量感 比标准大的445 小的11
def main():
    sources = mdb.style_source.find({})
    # true_count = 0
    # false_count = 0
    area_data = []
    for source in sources:
        face_struct = source['chin']
        left_eye = source['left_eye']
        right_eye = source['right_eye']

        face_points = [Point(p[0], p[1]) for p in face_struct]
        eye_dis_ratio = 100.0 / GetLineLength(
            Point((left_eye[3][0] + left_eye[0][0]) / 2.0, (left_eye[3][1] + left_eye[0][1]) / 2.0),
            Point((right_eye[0][0] + right_eye[3][0]) / 2.0, (right_eye[0][1] + right_eye[3][1]) / 2.0))
        S = 0
        for i in range(7):
            points = [face_points[i], face_points[i + 1], face_points[-i - 2], face_points[-i - 1]]
            area = GetAreaOfPolyGon(points)
            S += area
        face_points = [Point(p[0], p[1]) for p in [face_struct[7], face_struct[8], face_struct[9]]]
        S += GetAreaOfTriangle(face_points[0], face_points[1], face_points[2])
        mdb.style_source.update_one({"_id": source['_id']},
                                    {"$set": {"area": round(S * math.pow(eye_dis_ratio, 2), 2)}})
        # area_data.append(round(S * math.pow(eye_dis_ratio, 2), 2))
    # if round(S * math.pow(eye_dis_ratio, 2), 2) > 30179.07:
    #         true_count += 1
    #     else:
    #         false_count += 1
    # print(true_count, false_count)

    return area_data


if __name__ == '__main__':
    res = main()
