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
import tensorflow as tf

mdb = config.mdb
rdb = config.rdb
LabelList = ['TMKA', 'MLSS', 'QCJJ',
             'ZRYY', 'GYRM', 'ZXCZ',
             'LMMR', 'HLGY', 'XDMD']

tolerance = 0.3

sess = tf.InteractiveSession()

# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf_service.weight_variable([5, 5, 1, 32])
b_conv1 = tf_service.bias_variable([32])
h_conv1 = tf.nn.relu(tf_service.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf_service.max_pool_2x2(h_conv1)

W_conv2 = tf_service.weight_variable([5, 5, 32, 64])
b_conv2 = tf_service.bias_variable([64])
h_conv2 = tf.nn.relu(tf_service.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tf_service.max_pool_2x2(h_conv2)

W_fc1 = tf_service.weight_variable([7 * 7 * 64, 1024])
b_fc1 = tf_service.bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf_service.weight_variable([1024, 10])
b_fc2 = tf_service.bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


saver = tf.train.Saver()
saver.restore(sess, "resource/model/number_cnn.ckpt")


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
        sourceList = mdb.number_train_source.find()
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/model_number.html', LabelList=LabelList, source_list=source_list, pager=pager)

    def post(self):
        tf_service.number_train(0.01, 3000)
        return self.finish(base.rtjson())


class ModelNumberCNN(base.BaseHandler):
    def get(self):
        sourceList = mdb.number_train_source.find()
        source_list, pager = base.mongoPager(sourceList, self.input("pagenum", 1))
        return self.render('dxq_tf/model_number_cnn.html',source_list=source_list, pager=pager)

    def post(self):
        tf_service.number_cnn_train(0.0001, 3000)
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

        # res = tf_service.number_test(x_input)
        res = sess.run(y_conv, feed_dict={x: x_input})
        number_train_source = {
            '_id': src_id,
            'source': '/' + img_path,
            'train': '/' + train_path,
            'predict': np.argmax(res).tolist()
            # 'label':0 #编辑使用
        }
        mdb.number_train_source.insert(number_train_source)

        return self.finish(base.rtjson(num=str(np.argmax(res))))
