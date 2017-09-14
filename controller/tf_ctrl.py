# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''
from PIL import Image
import numpy as np

from common import base
import config
from service import tf_service, picture_service

# import model_variable
#
# number_softmax_sess = model_variable.number_softmax_sess
# number_softmax_x = model_variable.number_softmax_x
# number_softmax_y = model_variable.number_softmax_y
#
# style_softmax_sess = model_variable.style_softmax_sess
# style_softmax_x = model_variable.style_softmax_x
# style_softmax_y = model_variable.style_softmax_y
#
# number_cnn_sess = model_variable.number_cnn_sess
# number_cnn_x = model_variable.number_cnn_x
# number_cnn_y = model_variable.number_cnn_y
# number_cnn_keep_prob = model_variable.number_cnn_keep_prob
#
# style_cnn_sess = model_variable.style_cnn_sess
# style_cnn_x = model_variable.style_cnn_x
# style_cnn_y = model_variable.style_cnn_y
# style_cnn_keep_prob = model_variable.style_cnn_keep_prob

mdb = config.mdb
rdb = config.rdb
LabelList = ['TMKA', 'MLSS', 'QCJJ',
             'ZRYY', 'GYRM', 'ZXCZ',
             'LMMR', 'HLGY', 'XDMD']
FEATURES = [
    'chin',
    'left_eyebrow',
    'right_eyebrow',
    'nose_bridge',
    'nose_tip',
    'left_eye',
    'right_eye',
    'top_lip',
    'bottom_lip'
]

SenceAndOutline = {
    'TMKA': ['小', '曲'], 'MLSS': ['小', '中'], 'QCJJ': ['小', '直'],
    'ZRYY': ['中', '曲'], 'GYRM': ['中', '中'], 'ZXCZ': ['中', '直'],
    'LMMR': ['大', '曲'], 'HLGY': ['大', '中'], 'XDMD': ['大', '直']
}
tolerance = 0.3


class SourceIndex(base.BaseHandler):
    def get(self, typ):
        label = self.input("label", "all")
        match = {"type": typ} if label == 'all' else {"label": label, "type": typ}
        sourceList = mdb.style_source.find(match)
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/source_list.html', LabelList=LabelList, source_list=source_list,
                           pager=pager, label=label, typ=typ)


class SourceAdd(base.BaseHandler):
    def get(self):
        return self.render('dxq_tf/source_add.html', LabelList=LabelList)

    def post(self):
        face = self.input('face')  # Base64Img
        label = self.input('label')  # 标签
        typ = self.input('type')  # 类型
        print('label', label)
        print('type', typ)
        src_id = base.getRedisID('style_source')

        file_path = 'resource/style/origin/{}/{}.jpg'.format(label, src_id)
        # BaseImg 存入本地
        tf_service.saveBaseImg(face, file_path)
        print('saved')
        # 获取图片的关键点
        face_landmarks_dict = tf_service.face_landmarks(file_path)
        print('points')
        result = {}
        for feature in FEATURES:
            outline = face_landmarks_dict[feature]
            file_name = file_path.replace('origin', 'result/' + feature)
            tf_service.draw_points(points=outline, file_name=file_name)
            result[feature] = '/' + file_name
        if face_landmarks_dict != "Error":
            face_source = {
                '_id': int(src_id),
                'path': '/' + file_path,
                'label': label,
                'type': typ,
                'sense': SenceAndOutline[label][0],
                'outline': SenceAndOutline[label][1],
                'result': result
            }
            face_source.update(face_landmarks_dict)

            mdb.style_source.insert(face_source)

        return self.finish(base.rtjson(src_id=src_id))


# class SourceAdd2(base.BaseHandler):
#     def post(self):
#         import os
#         for label in LabelList:
#             pathDir = os.listdir('resource/' + label)
#             for sourceDir in pathDir:
#                 src_id = base.getRedisID('style_source')
#                 file_path = 'resource/style/origin/{}/{}.jpg'.format(label, src_id)
#
#                 fdir = file_path[:file_path.rfind('/')]
#                 if not os.path.exists(fdir):
#                     os.makedirs(fdir)
#                 if os.path.exists(file_path):
#                     continue
#                 Image.open('resource/{}/{}'.format(label, sourceDir)).save(file_path)
#
#                 # 获取图片的关键点
#                 face_landmarks_dict = tf_service.face_landmarks(file_path)
#                 result = {}
#                 for feature in FEATURES:
#                     outline = face_landmarks_dict[feature]
#                     file_name = file_path.replace('origin', 'result/' + feature)
#                     tf_service.draw_points(points=outline, file_name=file_name)
#                     result[feature] = '/' + file_name
#                 if face_landmarks_dict != "Error":
#                     face_source = {
#                         '_id': int(src_id),
#                         'path': '/' + file_path,
#                         'label': label,
#                         'type': "train",
#                         'sense': SenceAndOutline[label][0],
#                         'outline': SenceAndOutline[label][1],
#                         'result': result
#                     }
#                     face_source.update(face_landmarks_dict)
#
#                     mdb.style_source.insert(face_source)
#
#         return self.finish(base.rtjson())


class SourceEdit(base.BaseHandler):
    def get(self, src_id):
        face = mdb.face_train_source.find_one({"_id": int(src_id)})
        return self.render('dxq_tf/source_edit.html', face=face)


class UserIndex(base.BaseHandler):
    def get(self):
        return self.render('dxq_tf/user_list.html', LabelList=LabelList)

    def post(self):
        face = self.input("face")
        unknow_face_encoding, status = tf_service.face_encoding(face)
        if not status:
            return self.finish(base.rtjson(10002))

        known_faces, known_names = tf_service.get_know_face_encodings()

        results = tf_service.compare_faces(known_faces, unknow_face_encoding, top=3)
        name = []
        for res in results:
            if res[1] < tolerance:
                name.append(known_names[res[0]])
        return self.finish(base.rtjson(name=name))


class UserAdd(base.BaseHandler):
    def get(self):
        return self.render('dxq_tf/user_add.html')

    def post(self):
        face = self.input("face")
        name = self.input("name")
        face_encoding, status = tf_service.face_encoding(face)
        if not status:
            return self.finish(base.rtjson(10002))

        user_encoding = {
            '_id': base.getRedisID('user_encoding'),
            'name': name,
            'face_encoding': list(face_encoding)
        }
        mdb.user_encoding.insert(user_encoding)
        return self.finish(base.rtjson())


class ModelNumber(base.BaseHandler):
    def get(self):
        sourceList = mdb.number_train_source.find({"model": "softmax"})
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/model_number.html', source_list=source_list, pager=pager)

    def post(self):
        tf_service.number_train(0.01, 3000)
        return self.finish(base.rtjson())


class ModelNumberCNN(base.BaseHandler):
    def get(self):
        sourceList = mdb.number_train_source.find({"model": "cnn"})
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/model_number_cnn.html', source_list=source_list, pager=pager)

    def post(self):
        tf_service.number_cnn_train(0.0001, 3000)
        return self.finish(base.rtjson())


class ModelNumberTest(base.BaseHandler):
    def post(self, model):
        face = self.input('face')

        src_id = base.getRedisID('number_train_source')
        img_path = 'static/local/number/source_{}.jpg'.format(src_id)
        train_path = 'static/local/number/train_{}.jpg'.format(src_id)
        tf_service.saveBaseImg(face, img_path)

        img = Image.open(img_path)
        train = picture_service.cutSqure(img).convert("L").resize((28, 28), Image.ANTIALIAS)
        train.save(train_path)

        image = np.array(train).reshape(1, 784).astype(np.float32)
        x_input = np.multiply(image, 1.0 / 255.0)

        if model == 'softmax':
            res = number_softmax_sess.run(number_softmax_y, feed_dict={number_softmax_x: x_input})
        else:
            res = number_cnn_sess.run(number_cnn_y, feed_dict={number_cnn_x: x_input, number_cnn_keep_prob: 1})
        number_train_source = {
            '_id': src_id,
            'source': '/' + img_path,
            'train': '/' + train_path,
            'predict': np.argmax(res).tolist(),
            'model': model
            # 'label':0 #编辑使用
        }
        mdb.number_train_source.insert(number_train_source)

        return self.finish(base.rtjson(num=str(np.argmax(res))))


class ModelStyle(base.BaseHandler):
    def get(self):
        sourceList = mdb.style_train_source.find({"model": "softmax"})
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/model_style.html', source_list=source_list, pager=pager)

    def post(self):
        # tf_service.style_train(0.0001, 3000)
        tf_service.tz_train(0.001, 100000)
        return self.finish(base.rtjson())


class ModelStyleCNN(base.BaseHandler):
    def get(self):
        sourceList = mdb.style_train_source.find({"model": "cnn"})
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/model_style_cnn.html', source_list=source_list, pager=pager)

    def post(self):
        tf_service.style_cnn_train(0.0001, 3000)
        return self.finish(base.rtjson())


class ModelStyleTest(base.BaseHandler):
    def post(self, model):
        face = self.input('face')  # Base64Img
        label = "TMKA"  # 标签
        typ = "predict"  # 类型
        src_id = base.getRedisID('style_source')
        file_path = 'resource/style/origin/{}/{}.jpg'.format(label, src_id)
        # BaseImg 存入本地
        tf_service.saveBaseImg(face, file_path)
        # 获取图片的关键点
        face_landmarks_dict = tf_service.face_landmarks(file_path)
        result = {}
        for feature in FEATURES:
            outline = face_landmarks_dict[feature]
            file_name = file_path.replace('origin', 'result/' + feature)
            tf_service.draw_points(points=outline, file_name=file_name)
            result[feature] = '/' + file_name
        if face_landmarks_dict != "Error":
            face_source = {
                '_id': int(src_id),
                'path': '/' + file_path,
                'label': label,
                'type': typ,
                'sense': SenceAndOutline[label][0],
                'outline': SenceAndOutline[label][1],
                'result': result
            }
            face_source.update(face_landmarks_dict)

            mdb.style_source.insert(face_source)

        source = mdb.style_source.find_one({"_id": int(src_id)})
        file_path = source['result']['chin'].replace("/resource", "resource")
        img = Image.open(file_path)
        img2 = np.array(img.resize([28, 28]).convert("L")).reshape(1, 784).astype(np.float32)
        x_input = np.multiply(img2, 1.0 / 255.0)
        if model == 'softmax':
            res = style_softmax_sess.run(style_softmax_y, feed_dict={style_softmax_x: x_input})
        else:
            res = style_cnn_sess.run(style_cnn_y, feed_dict={style_cnn_x: x_input, style_cnn_keep_prob: 1})

        style = LabelList[np.argmax(res).tolist()]

        mdb.style_source.update_one({"_id": src_id}, {"$set": {"label": style, 'sense': SenceAndOutline[style][0],
                                                               'outline': SenceAndOutline[style][1]}})

        return self.finish(base.rtjson(style=style))
