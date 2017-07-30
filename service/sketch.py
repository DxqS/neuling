import cv2
import numpy as np
from landmarks import Position
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from PointCatch import position


def text(img):
    point_i = cv2.imread(img)
    cut_point = Position(point_i)
    face_outline, right_blow, left_blow, right_eye, left_eye, mouse, nose, ss = position(point_i)
    mouse_zone_up = np.zeros([4, 2])
    mouse_zone_up[0, :] = mouse[0, :]
    mouse_zone_up[1, :] = mouse[2, :] - 10
    mouse_zone_up[2, :] = mouse[4, :] - 10
    mouse_zone_up[3, :] = mouse[6, :]

    min_mouse_upY = min(mouse_zone_up[:, 1])
    max_mouse_upY = max(mouse_zone_up[:, 1])
    min_mouse_upX = min(mouse_zone_up[:, 0])
    max_mouse_upX = max(mouse_zone_up[:, 0])

    mouse_zone_bottom = mouse[6:13, :]
    min_mouse_bottomY = min(mouse_zone_bottom[:, 1])
    max_mouse_bottomY = max(mouse_zone_bottom[:, 1])
    min_mouse_bottomX = min(mouse_zone_bottom[:, 0])
    max_mouse_bottomX = max(mouse_zone_bottom[:, 0])

    min_left_blowX = min(left_blow[:, 0])
    max_left_blowX = max(left_blow[:, 0])
    min_left_blowY = min(left_blow[:, 1])
    max_left_blowY = max(left_blow[:, 1])

    min_right_blowX = min(right_blow[:, 0])
    max_right_blowX = max(right_blow[:, 0])
    min_right_blowY = min(right_blow[:, 1])
    max_right_blowY = max(right_blow[:, 1])

    MaxX = int(max(cut_point[:, 0]))
    MaxY = int(max(cut_point[:, 1]))
    MinX = int(min(cut_point[:, 0]))
    MinY = int(min(cut_point[:, 1]))

    I = Image.open(img)
    I = np.asarray(I, dtype="float64")
    [hei, wid, k] = I.shape
    rr = int(np.maximum(hei, wid) * 0.02)
    eps = 30  # 20+5*denoise*dn
    wd = 2  # whitening degree  base on log curves
    blur = 100
    alpha = 1
    I = whitening(I, wd)
    p = I
    q = np.zeros(I.shape)
    q[:, :, 0] = repix(I[:, :, 0], p[:, :, 0], rr, eps)
    q[:, :, 1] = repix(I[:, :, 1], p[:, :, 1], rr, eps)
    q[:, :, 2] = repix(I[:, :, 2], p[:, :, 2], rr, eps)
    FilterI = Image.fromarray(np.uint8(q))

    Box = (MinX - 30, MinY - 30, MaxX + 30, MaxY + 30)
    CutPII = FilterI.crop(Box)
    II = np.asarray(FilterI, dtype="float64")
    q = np.zeros(II.shape)
    q[:, :, 0] = 255
    q[:, :, 1] = 255
    q[:, :, 2] = 255
    Back = Image.fromarray(np.uint8(q))
    Back.paste(CutPII, Box)

    SketchI = Back.convert('L')
    # SketchI.show()
    SketchI_copy = SketchI.copy()
    SketchI_copy = ImageOps.invert(SketchI_copy)
    k0, b0 = line(mouse_zone_up)
    k1, b1 = line(mouse_zone_bottom)
    k2, b2 = line(left_blow)
    k3, b3 = line(right_blow)
    for i in range(blur):
        SketchI_copy = SketchI_copy.filter(ImageFilter.GaussianBlur)
    for x in range(MinX, MaxX):
        for y in range(MinY, MaxY):
            if y >= min_mouse_bottomY and y <= max_mouse_bottomY and x >= min_mouse_bottomX and x <= max_mouse_bottomX:
                ff = (y - k1 * x + b1) > 0
                sum = np.sum(ff)
                if sum == 0:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.001))
                else:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.1))
            elif y >= min_mouse_upY and y <= max_mouse_upY and x >= min_mouse_upX and x <= max_mouse_upX:
                ff = (y - k0 * x + b0) < 0
                sum = np.sum(ff)
                if sum == 0:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.001))
                else:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.1))
            elif y >= min_left_blowY and y <= max_left_blowY and x >= min_left_blowX and x <= max_left_blowX:
                ff = (y - k2 * x + b2) < 0
                sum = np.sum(ff)
                if sum == 0:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.01))
                else:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.1))
            elif y >= min_right_blowY and y <= max_right_blowY and x >= min_right_blowX and x <= max_right_blowX:
                ff = (y - k3 * x + b3) < 0
                sum = np.sum(ff)
                if sum == 0:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.01))
                else:
                    a = SketchI.getpixel((x, y))
                    b = SketchI_copy.getpixel((x, y))
                    SketchI.putpixel((x, y), dodge(a, b, alpha, 1.1))
            else:
                a = SketchI.getpixel((x, y))
                b = SketchI_copy.getpixel((x, y))
                SketchI.putpixel((x, y), dodge(a, b, alpha, 1.1))

    I = np.asarray(SketchI, dtype="float64")
    I = PGStop(cut_point, I, 0)
    I = PGS1(cut_point, I, 3)
    I = Image.fromarray(np.uint8(I))
    sharpness = ImageEnhance.Sharpness(I)
    TypeI = sharpness.enhance(1)
    contrast = ImageEnhance.Contrast(TypeI)
    TypeI = contrast.enhance(2)
    for i in range(20):
        contrast = ImageEnhance.Contrast(TypeI)
        TypeI = contrast.enhance(1.1)
        mean = IA(TypeI)
        if mean > 3.5:
            print mean
            break

    return TypeI


def dodge(a, b, alpha, beta):
    return min(int(beta * a * 255 / (256 - b * alpha)), 255)


def IA(I):
    copy_I = np.asarray(I, dtype="float64")
    copy_I = 1 - copy_I / 255
    [hei, wid] = copy_I.shape
    sum = copy_I.sum()
    mean = 1000*sum / (wid * hei)
    return mean


def PGS1(cut_point, I, t):
    MaxX = int(max(cut_point[:, 0]))
    MaxY = int(max(cut_point[:, 1]))
    MinX = int(min(cut_point[:, 0]))
    MinY = int(min(cut_point[:, 1]))
    Point1 = cut_point[0:-2, :]
    for i in range(15):
        Point1 = adj(Point1)
    for i in range(t):
        Point1 = add(Point1)
    [wid1, hei1] = Point1.shape
    k1 = (Point1[1:, 1] - Point1[0:-1, 1]) / (Point1[1:, 0] + .0001 - Point1[0:-1, 0])
    c = np.zeros([2, 2])
    b1 = np.zeros([wid1 - 1])
    for i in range(wid1 - 1):
        c[0, :] = Point1[i, :]
        c[1, :] = Point1[i + 1, :]
        b1[i] = np.linalg.det(c) / (Point1[i + 1, 0] + .0001 - Point1[i, 0])
    for x in range(MinX - 30, MaxX + 30):
        for y in range(MinY - 30, MaxY + 30):
            s = f1(x, y, k1, b1)
            if s == 0:
                I[y, x] = 255
    return I


def line(Point):
    Point1 = Point
    for i in range(2):
        Point1 = add(Point1)
    [wid1, hei1] = Point1.shape
    k1 = (Point1[1:, 1] - Point1[0:-1, 1]) / (Point1[1:, 0] - Point1[0:-1, 0])
    c = np.zeros([2, 2])
    b1 = np.zeros([wid1 - 1])
    for i in range(wid1 - 1):
        c[0, :] = Point1[i, :]
        c[1, :] = Point1[i + 1, :]
        b1[i] = (np.linalg.det(c)) / (Point1[i + 1, 0] - Point1[i, 0])
    return k1, b1


def PGStop(Point, I, t):
    MaxX = int(max(Point[:, 0]))
    MaxY = int(max(Point[:, 1]))
    MinX = int(min(Point[:, 0]))
    MinY = int(min(Point[:, 1]))
    Point2 = np.zeros([4, 2])
    Point2[0, :] = Point[-3, :]
    Point2[1, :] = Point[-2, :]
    Point2[2, :] = Point[-1, :]
    Point2[3, :] = Point[0, :]
    for i in range(t):
        Point2 = add(Point2)
    [wid2, hei2] = Point2.shape
    k2 = (Point2[1:, 1] - Point2[0:-1, 1]) / (Point2[1:, 0] + .0001 - Point2[0:-1, 0])
    c = np.zeros([2, 2])
    b2 = np.zeros([wid2 - 1])
    for i in range(wid2 - 1):
        c[0, :] = Point2[i, :]
        c[1, :] = Point2[i + 1, :]
        b2[i] = np.linalg.det(c) / (Point2[i + 1, 0] + .0001 - Point2[i, 0])
    for x in range(MinX - 30, MaxX + 30):
        for y in range(MinY - 30, MaxY + 30):
            s = f2(x, y, k2, b2)
            if s == 0:
                I[y, x] = 255
    return I


def adj(Point):
    Point[0, 0] = Point[1, 0] - .1
    Point[-1, 0] = Point[-2, 0] + .1
    k = (Point[1:, 1] - Point[0:-1, 1]) / (Point[1:, 0] + .1 - Point[0:-1, 0])
    Kcheck = k[1:] - k[0:-1] <= 0
    for i in range(len(Kcheck)):
        if ~Kcheck[i]:
            Point[i + 1, :] = (Point[i, :] + Point[i + 2, :]) / 2
    return (Point)


def add(Point):
    [wid, hei] = Point.shape
    NewP = np.zeros([wid + 1, hei])
    for i in range(wid + 1):
        if i == 0:
            NewP[i, :] = Point[i, :]
        if i == wid:
            NewP[i, :] = Point[wid - 1, :]
        if i > 0 and i < wid:
            NewP[i, :] = .5 * Point[i - 1, :] + .5 * Point[i, :]
    return NewP


def f1(x, y, k, b):
    ff = (y - k * x + b) > 0
    sum = np.sum(ff)
    if sum == 0:
        return 1
    else:
        return 0


def f2(x, y, k, b):
    ff = (y - k * x + b) < 0
    sum = np.sum(ff)
    if sum == 0:
        return 1
    else:
        return 0


def whitening(img, wd):
    R = img[:, :, 0] / 255
    G = img[:, :, 1] / 255
    B = img[:, :, 2] / 255
    k = np.zeros(img.shape)
    k[:, :, 0] = np.log(R * (wd - 1) + 1) / np.log(wd)
    k[:, :, 1] = np.log(G * (wd - 1) + 1) / np.log(wd)
    k[:, :, 2] = np.log(B * (wd - 1) + 1) / np.log(wd)
    k = k * 255
    return k


def repix(I, p, r, eps):
    [hei, wid] = I.shape
    N = Ssum(np.ones([hei, wid]), r)
    mean_I = Ssum(I, r) / N
    mean_p = Ssum(p, r) / N
    mean_Ip = Ssum(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = Ssum(I * I, r) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = Ssum(a, r) / N
    mean_b = Ssum(b, r) / N
    q = mean_a * I + mean_b
    return q


def Ssum(img, r):
    [hei, wid] = img.shape
    imDst = np.zeros(img.shape)
    imCum = img.cumsum(axis=0)
    imDst[0:r + 1, :] = imCum[r:2 * r + 1, :]
    imDst[r + 1:hei - r, :] = imCum[2 * r + 1:hei, :] - imCum[0:hei - 2 * r - 1, :]
    imDst[hei - r:hei, :] = np.kron(np.ones((r, 1)), imCum[hei - 1, :]) - imCum[hei - 2 * r - 1:hei - r - 1, :]

    imCum = imDst.cumsum(axis=1)
    imDst[:, 0:r + 1] = imCum[:, r:2 * r + 1]
    imDst[:, r + 1:wid - r] = imCum[:, 2 * r + 1:wid] - imCum[:, 0:wid - 2 * r - 1]
    imDst[:, wid - r:wid] = np.kron(np.ones((r, 1)), imCum[:, wid - 1]).transpose() - imCum[:,
                                                                                      wid - 2 * r - 1:wid - r - 1]

    return imDst
