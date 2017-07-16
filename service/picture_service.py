# coding:utf-8
'''
Created on 2017/7/16.

@author: Dxq
'''
import base64
import os
from common import tools

IMG_TYPE = ['png', 'jpeg', 'jpg']


def Base64_to_File(baseImg, file_dir):
    for typ in IMG_TYPE:
        replace_str = "data:image/{};base64,".format(typ)
        baseImg = baseImg.replace(replace_str, "")

        file_path = 'static/download/{}/{}.jpg'.format(file_dir, tools.createNoncestr())
        fdir = file_path[:file_path.rfind('/')]
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        imgdata = base64.b64decode(baseImg)
        with open(file_path, 'wb') as f:
            f.write(imgdata)
        return file_path
