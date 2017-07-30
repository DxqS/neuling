# coding:utf-8
'''
Created on 2017/7/16.

@author: Dxq
'''
import base64
import os
from PIL import Image
import math
import time

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

    r, g, b, a = poster_back.convert("RGBA").split()
    white_back.paste(poster_back, (0, 0, size[0], size[1]), mask=a)
    white_back.save('static/pic/3.jpg')
    return '/static/pic/3.jpg'


def circle(img):
    # 会压缩成正方形再处理
    ima = img.convert("RGBA")
    size = ima.size
    r2 = min(size[0], size[1])

    if size[0] != size[1]:
        ima = ima.resize((r2, r2), Image.ANTIALIAS)
    imb = Image.new('RGBA', (r2, r2), (255, 255, 255, 0))
    pima = ima.load()
    pimb = imb.load()
    r = float(r2 / 2)
    for i in range(r2):
        for j in range(r2):
            lx = abs(i - r + 0.5)
            ly = abs(j - r + 0.5)
            l = pow(lx, 2) + pow(ly, 2)
            if l <= pow(r, 2):
                pimb[i, j] = pima[i, j]
    return imb


def cutSqure(img):
    # 居中裁剪正方形
    ts = time.time()
    size = img.size
    s_max = max(size[0], size[1])
    s_min = min(size[0], size[1])

    start_x = (s_max - s_min) / 2 if s_max == size[0] else 0
    start_y = (s_max - s_min) / 2 if s_max == size[1] else 0
    head_box = (start_x, start_y, start_x + s_min, start_y + s_min)

    img = img.crop(head_box)
    print time.time()-ts
    return img


def love(img):
    # 会压缩成正方形再处理
    ima = img.convert("RGBA")
    size = ima.size
    r2 = min(size[0], size[1])

    if size[0] != size[1]:
        ima = ima.resize((r2, r2), Image.ANTIALIAS)
    imb = Image.new('RGBA', (r2, r2), (255, 255, 255, 0))
    pima = ima.load()
    pimb = imb.load()
    r = float(r2 / 2)
    for i in range(r2):
        for j in range(r2):
            # lx = abs(i - r + 0.5)
            # ly = abs(j - r + 0.5)

            l = j - math.sin(i)
            if l <= 0:
                pimb[i, j] = pima[i, j]
    return imb


