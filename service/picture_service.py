# coding:utf-8
'''
Created on 2017/7/16.

@author: Dxq
'''
import base64
import os
from PIL import Image

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


def Poster_Add(img1, img2, position):
    poster_back = Image.open(img1)
    size = poster_back.size
    white_back = Image.new("RGBA", size, color=(255, 255, 255, 1))

    add_img = Image.open(img2)
    self_wid, self_hei = add_img.size
    show_wid = position[2]
    if self_hei > self_wid:
        r = float(self_hei * show_wid) / self_wid
        self_pic = add_img.resize((show_wid, int(r)), Image.ANTIALIAS)
        box = (position[0], position[1], position[0] + show_wid, position[1] + int(r))
    else:
        r = float(self_wid * show_wid) / self_hei
        self_pic = add_img.resize((int(r), show_wid), Image.ANTIALIAS)
        box = (position[0], position[1], position[0] + int(r), position[1] + show_wid)
    white_back.paste(self_pic, box)

    r, g, b, a = poster_back.split()
    white_back.paste(poster_back, (0, 0, size[0], size[1]), mask=a)
    white_back.save('3.jpg')
    return True
