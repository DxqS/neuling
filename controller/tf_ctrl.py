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

mdb = config.mdb
rdb = config.rdb
LabelList = ['TMKA', 'MLSS', 'QCJJ',
             'ZRYY', 'GYRM', 'ZXCZ',
             'LMMR', 'HLGY', 'XDMD']

tolerance = 0.3


class SourceIndex(base.BaseHandler):
    def get(self):
        label = self.input("label", "all")
        match = {} if label == 'all' else {"label": label}
        sourceList = mdb.face_train_source.find(match)
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/source_list.html', LabelList=LabelList, source_list=source_list,
                           pager=pager, label=label)


class SourceAdd(base.BaseHandler):
    def get(self):
        return self.render('dxq_tf/source_add.html', LabelList=LabelList)

    def post(self):
        face = self.input('face')
        label = self.input('label')
        src_id = base.getRedisID('face_train_source')
        path = tf_service.Add_Face_to_Source(face, label, src_id)
        tf_service.Add_Face_DB(path, label, src_id)
        return self.finish(base.rtjson())


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
        return self.render('dxq_tf/model_number.html', LabelList=LabelList)

    def post(self):
        tf_service.number_train(0.01, 3000)
        return self.finish(base.rtjson())


class ModelTest(base.BaseHandler):
    def post(self):
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
        res = tf_service.number_test(x_input)
        number_train_source = {
            '_id': src_id,
            'source': img_path,
            'train': train_path,
            'predict': np.argmax(res).tolist()
            # 'label':0 #编辑使用
        }
        mdb.number_train_source.insert(number_train_source)

        return self.finish(base.rtjson(num=str(np.argmax(res))))
