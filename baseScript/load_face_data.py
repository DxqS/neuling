# coding:utf-8
'''
Created on 2017/9/7.

@author: chk01
'''
import numpy as np
import os
import scipy.io as scio
from PIL import Image

LabelToCode = {
    'TMKA': [1, 0, 0, 0, 0, 0, 0, 0, 0], 'MLSS': [0, 1, 0, 0, 0, 0, 0, 0, 0], 'QCJJ': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'ZRYY': [0, 0, 0, 1, 0, 0, 0, 0, 0], 'GYRM': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'ZXCZ': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'LMMR': [0, 0, 0, 0, 0, 0, 1, 0, 0], 'HLGY': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'XDMD': [0, 0, 0, 0, 0, 0, 0, 0, 1],
}


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
        img2 = np.array(img.resize([28, 28]).convert("L")).reshape(1, 784).astype(np.float32)
        image = np.multiply(img2, 1.0 / 255.0)
        x[i + 1:] = image
        y[i + 1:] = LabelToCode[source['label']]
    scio.savemat(file_name, {'X': x, 'Y': y})
    return True


if __name__ == "__main__":
    load_data_mat('face_data.mat')
    # data = scio.loadmat('face_data.mat')
