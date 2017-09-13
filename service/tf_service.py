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
import time
from PIL import ImageDraw, Image
import scipy.io as scio
import random

from common import tools
import config

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


def draw_points(points, file_name):
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
    res = region.resize((300, 300), Image.ANTIALIAS).convert("L")
    res.save(file_name)
    return True


def get_know_face_encodings():
    known_faces = []
    known_names = []
    for rec in mdb.user_encoding.find():
        known_faces.append(np.array(rec['face_encoding']))
        known_names.append(rec['name'])

    return known_faces, known_names


# def load_data_mat(file_name):
#     if os.path.exists(file_name):
#         return True
#     train_source = mdb.face_train_source.find()
#     num = train_source.count()
#     x = np.zeros([num, 784])
#     y = np.zeros([num, 9])
#     for i, source in enumerate(train_source):
#         out_line = source['result']['chin']
#         img = Image.open('..' + out_line)
#         x[i + 1:] = np.array(img.resize([28, 28]).convert("L")).reshape(1, 784)
#         y[i + 1:] = LabelToCode[source['label']]
#     scio.savemat(file_name, {'X': x, 'Y': y})
#     return True


def get_random_block_from_data(data, batch_size):
    num = len(data['X'])
    randomlist = random.sample(range(num - batch_size), batch_size)
    return [data['X'][i] for i in randomlist], [data['Y'][i] for i in randomlist]


def number_train(learning_rate, train_epochs):
    # 数据集已经处理完毕存resource/mnist_data.mat后才能正常使用
    sess = tf.InteractiveSession()
    data = scio.loadmat('resource/mnist_data.mat')

    # 限定命名空间
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)
    # W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
    # b = tf.Variable(tf.constant(0.1, shape=[10]))
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]))

    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    with tf.name_scope('cross_entropy'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar("cross_entropy", cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('resource/summary/number/softmax/train', sess.graph)
    tf.global_variables_initializer().run()
    for step in range(train_epochs):
        xs_batch, ys_batch = get_random_block_from_data(data, 100)
        train_step.run({x: xs_batch, y_: ys_batch})
        summary = sess.run(merged, feed_dict={x: xs_batch, y_: ys_batch})
        train_writer.add_summary(summary, step)
        if step % 100 == 0:
            print(accuracy.eval(feed_dict={x: xs_batch, y_: ys_batch}))

    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "resource/model/number/softmax/model.ckpt")
    return True


def style_train(learning_rate, train_epochs):
    # 数据集已经处理完毕存resource/face_data.mat后才能正常使用
    sess = tf.InteractiveSession()
    data = scio.loadmat('resource/face_data.mat')

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 9])

    W = tf.Variable(tf.zeros([784, 9]))
    b = tf.Variable(tf.zeros([9]))
    tf.global_variables_initializer().run()

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for step in range(train_epochs):
        xs_batch, ys_batch = get_random_block_from_data(data, 100)
        train_step.run({x: xs_batch, y_: ys_batch})
        if step % 100 == 0:
            print(accuracy.eval(feed_dict={x: xs_batch, y_: ys_batch}))

    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "resource/model/style/softmax/model.ckpt")
    return True


def tz_train(learning_rate, train_epochs):
    # 数据集已经处理完毕存resource/tz_data.mat后才能正常使用
    sess = tf.InteractiveSession()
    data = scio.loadmat('resource/tz_data.mat')

    # 限定命名空间
    with tf.name_scope("input"):
        x1 = tf.placeholder(tf.float32, [None, 1], name='x1-input')
        x2 = tf.placeholder(tf.float32, [None, 1], name='x2-input')
        y_ = tf.placeholder(tf.float32, [None, 3], name='y-input')

    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.zeros([1, 3]))
        W2 = tf.Variable(tf.zeros([1, 3]))
        W3 = tf.Variable(tf.zeros([6, 3]))
    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.zeros([3]))
        b2 = tf.Variable(tf.zeros([3]))
        b3 = tf.Variable(tf.zeros([3]))

    y1 = tf.nn.softmax(tf.matmul(x1, W1) + b1)
    y2 = tf.nn.softmax(tf.matmul(x2, W2) + b2)
    y = tf.nn.softmax(tf.matmul(tf.reshape(tf.stack([y1, y2], 1), [-1, 6]), W3) + b3)
    with tf.name_scope('cross_entropy'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar("cross_entropy", cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('resource/summary/tz/softmax/train', sess.graph)
    tf.global_variables_initializer().run()
    for step in range(train_epochs):
        xs_batch, ys_batch = get_random_block_from_data(data, 100)
        train_step.run(
            {x1: np.array([[x[0]] for x in xs_batch]), x2: np.array([[x[1]] for x in xs_batch]), y_: ys_batch})
        summary = sess.run(merged, feed_dict={x1: np.array([[x[0]] for x in xs_batch]),
                                              x2: np.array([[x[1]] for x in xs_batch]), y_: ys_batch})
        train_writer.add_summary(summary, step)
        if step % 100 == 0:
            # ww = sess.run(W, feed_dict={x: xs_batch, y_: ys_batch})
            print(accuracy.eval(
                feed_dict={x1: np.array([[x[0]] for x in xs_batch]), x2: np.array([[x[1]] for x in xs_batch]),
                           y_: ys_batch}))

    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "resource/model/tz/softmax/model.ckpt")
    return True


###################################################CNN模型###############################################

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def number_cnn_train(learning_rate, train_epochs):
    ts = time.time()
    sess = tf.InteractiveSession()
    data = scio.loadmat('resource/mnist_data.mat')

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('x-input', x_image, 10)

    with tf.name_scope("W_conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
    with tf.name_scope("b_conv1"):
        b_conv1 = bias_variable([32])

    with tf.name_scope("layer1"):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    # feature1 = tf.reshape(h_pool1, [-1, 14, 14, 1], name='h_pool1')
    # tf.summary.image('feature1', feature1, 10)

    with tf.name_scope("W_conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
    with tf.name_scope("b_conv2"):
        b_conv2 = bias_variable([64])

    with tf.name_scope("layer2"):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
    # feature2 = tf.reshape(h_pool2, [-1, 7, 7, 1], name='h_pool2')
    # tf.summary.image('feature2', feature2, 10)

    with tf.name_scope("W_fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
    with tf.name_scope("b_fc1"):
        b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("W_fc2"):
        W_fc2 = weight_variable([1024, 10])
    with tf.name_scope("b_fc2"):
        b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.name_scope("cross_entropy"):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('resource/summary/number/cnn/train', sess.graph)
    tf.global_variables_initializer().run()

    for step in range(train_epochs):
        xs_batch, ys_batch = get_random_block_from_data(data, 50)
        summary = sess.run(merged, feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 1.0})
        train_writer.add_summary(summary, step)
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 1.0})
            print("step %d,training accuracy %g" % (step, train_accuracy))
        train_step.run(feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 0.5})
    # print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "resource/model/number/cnn/model.ckpt")
    print(time.time() - ts)
    return True


def style_cnn_train(learning_rate, train_epochs):
    ts = time.time()
    sess = tf.InteractiveSession()
    data = scio.loadmat('resource/face_data.mat')

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 9])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 9])
    b_fc2 = bias_variable([9])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run()

    for i in range(train_epochs):
        xs_batch, ys_batch = get_random_block_from_data(data, 50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 1.0})
            print("step %d,training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: xs_batch, y_: ys_batch, keep_prob: 0.5})
    # print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, "resource/model/style/cnn/model.ckpt")
    print(time.time() - ts)
    return True


###################################################CNN模型###############################################

def extract_image(face):
    return 1


def saveBaseImg(baseImg, file_path='static/local/temporary/1.jpg'):
    for typ in IMG_TYPE:
        replace_str = "data:image/{};base64,".format(typ)
        baseImg = baseImg.replace(replace_str, "")

    fdir = file_path[:file_path.rfind('/')]
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    imgdata = base64.b64decode(baseImg)
    with open(file_path, 'wb') as f:
        f.write(imgdata)
    return file_path
