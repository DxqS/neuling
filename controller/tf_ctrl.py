# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''

from common import base
import config
from service import tf_service

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


class TrainIndex(base.BaseHandler):
    def get(self):
        return self.render('dxq_tf/train_index.html', LabelList=LabelList)

    def post(self):
        tf_service.train(0.05, 100)
        return self.finish(base.rtjson())


class TrainTest(base.BaseHandler):
    def post(self):
        face = self.input('face')
        src_id = base.getRedisID('face_test_source')
        path = tf_service.Add_Face_to_Source(face, 'TEST', src_id)
        tf_service.Add_Face_DB(path, 'TEST', src_id)

        print(tf_service.tt(src_id))

        return self.finish(base.rtjson())
