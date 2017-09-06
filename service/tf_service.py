# coding:utf-8
'''
Created on 2017/8/28.

@author: Dxq
'''
import os
import base64
import face_recognition
import tensorflow as tf
import numpy as np
from PIL import ImageDraw, Image
from common import tools
import config
import scipy.io as scio

LabelToCode = {
    'TMKA': [1, 0, 0, 0, 0, 0, 0, 0, 0], 'MLSS': [0, 1, 0, 0, 0, 0, 0, 0, 0], 'QCJJ': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'ZRYY': [0, 0, 0, 1, 0, 0, 0, 0, 0], 'GYRM': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'ZXCZ': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'LMMR': [0, 0, 0, 0, 0, 0, 1, 0, 0], 'HLGY': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'XDMD': [0, 0, 0, 0, 0, 0, 0, 0, 1],
}
mdb = config.mdb
rdb = config.rdb
IMG_TYPE = ['png', 'jpeg', 'jpg']
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


def Add_Face_to_Source(baseImg, label, sid):
    for typ in IMG_TYPE:
        replace_str = "data:image/{};base64,".format(typ)
        baseImg = baseImg.replace(replace_str, "")

    file_path = 'PreHandle/Source/{}/{}.jpg'.format(label, sid)
    fdir = file_path[:file_path.rfind('/')]
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    imgdata = base64.b64decode(baseImg)
    with open(file_path, 'wb') as f:
        f.write(imgdata)
    return file_path


def face_encoding(baseImg):
    status = True
    for typ in IMG_TYPE:
        replace_str = "data:image/{};base64,".format(typ)
        baseImg = baseImg.replace(replace_str, "")

    file_path = 'PreHandle/Temporary/{}.jpg'.format(tools.uniqueName())
    fdir = file_path[:file_path.rfind('/')]
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    imgdata = base64.b64decode(baseImg)
    with open(file_path, 'wb') as f:
        f.write(imgdata)

    load_image = face_recognition.load_image_file(file_path)
    try:
        res_face_encoding = face_recognition.face_encodings(load_image)[0]
    except:
        status = False
        res_face_encoding = None
    os.remove(file_path)
    return res_face_encoding, status


def compare_faces(known_faces, unknow_face_encoding, top=3):
    res = [(i, t) for i, t in enumerate(list(face_distance(known_faces, unknow_face_encoding)))]
    res.sort(key=lambda x: x[1])
    return res[:top]
    # return face_recognition.compare_faces(known_faces, unknow_face_encoding, tolerance)


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def face_landmarks(face_image):
    image = face_recognition.load_image_file(face_image)
    face_landmarks_list = face_recognition.face_landmarks(image)

    return face_landmarks_list[0]


def drawPoints(points, file_name):
    file_dir = file_name[:file_name.rfind('/')]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    min_x = min(x) - 5
    min_y = min(y) - 5
    max_x = max(x) + 5
    max_y = max(y) + 5
    wid = max(max_y - min_y, max_x - min_x)

    pil_image = Image.open('static/images/1.jpg')
    d = ImageDraw.Draw(pil_image)
    d.line(points, width=5)
    region = pil_image.crop((min_x, min_y, min_x + wid, min_y + wid))
    res = region.resize((300, 300), Image.ANTIALIAS)
    res.save(file_name)
    return True


def Add_Face_DB(path, label, src_id):
    face_landmarks_dict = face_landmarks(path)  # 暂时只画轮廓
    result = {}
    for feature in FEATURES:
        outline = face_landmarks_dict[feature]
        file_name = path.replace('Source', 'Result/' + feature)
        drawPoints(points=outline, file_name=file_name)
        result[feature] = '/' + file_name
    if face_landmarks_dict != "Error":
        table_name = 'face_train_source' if label != 'TEST'else 'face_test_source'

        face_source = {
            '_id': int(src_id),
            'path': '/' + path,
            'label': label,
            'type': 'train',
            'result': result
        }
        face_source.update(face_landmarks_dict)
        mdb[table_name].insert(face_source)
    return True


def get_know_face_encodings():
    known_faces = []
    known_names = []
    for rec in mdb.user_encoding.find():
        known_faces.append(np.array(rec['face_encoding']))
        known_names.append(rec['name'])

    return known_faces, known_names


def load_data_mat(file_name):
    if os.path.exists(file_name):
        return True
    train_source = mdb.face_train_source.find()
    num = train_source.count()
    x = np.zeros([num, 784])
    y = np.zeros([num, 9])
    for i, source in enumerate(train_source):
        out_line = source['result']['chin']
        img = Image.open('..' + out_line)
        x[i + 1:] = np.array(img.resize([28, 28]).convert("L")).reshape(1, 784)
        y[i + 1:] = LabelToCode[source['label']]
    scio.savemat(file_name, {'X': x, 'Y': y})
    return True


def get_random_block_from_data(data, batch_size):
    num = len(data['X'])
    start_index = np.random.randint(0, num - batch_size)
    return data['X'][start_index:(start_index + batch_size)], data['Y'][start_index:(start_index + batch_size)]


def train(learning_rate, train_epochs):
    sess = tf.InteractiveSession()
    load_data_mat('resource/mnist_data.mat')
    data = scio.loadmat('resource/mnist_data.mat')

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
    # b = tf.Variable(tf.constant(0.1, shape=[10]))
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    tf.global_variables_initializer().run()

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    for step in range(train_epochs):
        xs_batch, ys_batch = get_random_block_from_data(data, 100)
        batch = mnist.train.next_batch(100)
        if step == 0:
            print('my', xs_batch)
            print('mnist', batch[0])
            print('my', type(xs_batch))
            print('mnist', type(batch[0]))
            print('my', xs_batch[0][0])
            print('mnist', batch[0][0][0])
            print('my', type(xs_batch[0][0]))
            print('mnist', type(batch[0][0][0]))
            print('my', xs_batch.shape)
            print('mnist', batch[0].shape)

        train_step.run({x: batch[0], y_: batch[1]})
        if step % 20 == 0:
            ww = sess.run(W)
            print(step, ww)
            print(ww.shape)
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "resource/model/face.ckpt")
    return True


def tt(sid):
    test_source = mdb.face_test_source.find_one({"_id": int(sid)})
    x_input = np.zeros([1, 34])
    xs = []
    for point in test_source['chin']:
        xs.extend(point)

    x_input[1:] = np.transpose(np.array(xs))

    # 定义变量
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    saver = tf.train.Saver()
    saver.restore(sess, "resource/model/face.ckpt")
    res = sess.run(y, feed_dict={x: x_input})
    return res
