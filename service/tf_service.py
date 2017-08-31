# coding:utf-8
'''
Created on 2017/8/28.

@author: Dxq
'''
import os
import base64
import face_recognition
import numpy as np
from PIL import ImageDraw, Image
from common import tools
import config

mdb = config.mdb
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

    biden_image = face_recognition.load_image_file(file_path)
    try:
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    except:
        status = False
        biden_face_encoding = None
    os.remove(file_path)
    return biden_face_encoding, status


def compare_faces(known_faces, unknow_face_encoding, tolerance=0.6):
    tt = face_distance(known_faces, unknow_face_encoding)
    print([(i, t) for i, t in enumerate(list(tt))].sort(key=lambda x: x[1]))
    return list(face_distance(known_faces, unknow_face_encoding) <= tolerance)
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
        face_train_source = {
            '_id': int(src_id),
            'path': '/' + path,
            'label': label,
            'type': 'train',
            'result': result
        }
        face_train_source.update(face_landmarks_dict)
        mdb.face_train_source.insert(face_train_source)
    return True
