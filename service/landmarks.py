import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def Position(im):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    ss = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    k1 = ss[0:17, :]
    kk = np.zeros([19, 2])
    kk[0:17, :] = k1
    kk[17, :] = ss[18, :]
    kk[18, :] = ss[25, :]
    c = np.zeros([19, 2])

    # c[0, :] = [kk[0, 0] * 1.05, kk[0, 1]]
    # c[1, :] = [kk[1, 0] * 1.05, kk[1, 1]]
    # c[2, :] = [kk[2, 0], kk[2, 1]]
    # c[3, :] = [kk[3, 0] * 1.01, kk[3, 1]]
    # c[4, :] = [kk[4, 0] * 1.01, kk[4, 1]]
    # c[5, :] = [kk[5, 0] * 1.01, kk[5, 1] * .95]
    # c[6, :] = [kk[6, 0] * 1.01, kk[6, 1] * .95]
    # c[7, :] = [kk[7, 0] * 1.01, kk[7, 1] * .95]
    # c[8, :] = [kk[8, 0], kk[8, 1] * .97]  # mid point
    # c[9, :] = [kk[9, 0], kk[9, 1] * .97]
    # c[10, :] = [kk[10, 0], kk[10, 1]*.97]
    # c[11, :] = [kk[11, 0], kk[11, 1]*.97]
    # c[12, :] = [kk[12, 0] * .98, kk[12, 1] * .98]
    # c[13, :] = [kk[13, 0] * .98, kk[13, 1]]
    # c[14, :] = [kk[14, 0] * .98, kk[14, 1]]
    # c[15, :] = [kk[15, 0] * .95, kk[15, 1]]
    # c[16, :] = [kk[16, 0] * .95, kk[16, 1]]
    # c[17, :] = [kk[16, 0] * .9, kk[17, 1] * .97]
    # c[18, :] = [kk[0, 0] * 1.2, kk[18, 1] * .97]
    c[0, :] = [kk[0, 0] + 20, kk[0, 1]]
    c[1, :] = [kk[1, 0] + 20, kk[1, 1]]
    c[2, :] = [kk[2, 0], kk[2, 1]]
    c[3, :] = [kk[3, 0] + 3, kk[3, 1] + 2]
    c[4, :] = [kk[4, 0] + 3, kk[4, 1]]
    c[5, :] = [kk[5, 0] + 5, kk[5, 1] - 17]
    c[6, :] = [kk[6, 0] + 5, kk[6, 1] - 17]
    c[7, :] = [kk[7, 0] + 5, kk[7, 1] - 17]
    c[8, :] = [kk[8, 0], kk[8, 1]-5]  # mid point
    c[9, :] = [kk[9, 0], kk[9, 1]-15]
    c[10, :] = [kk[10, 0], kk[10, 1]]
    c[11, :] = [kk[11, 0], kk[11, 1]]
    c[12, :] = [kk[12, 0] - 14, kk[12, 1] - 4]
    c[13, :] = [kk[13, 0]-10, kk[13, 1]]
    c[14, :] = [kk[14, 0]-10, kk[14, 1]]
    c[15, :] = [kk[15, 0] - 20, kk[15, 1]]
    c[16, :] = [kk[16, 0] - 20, kk[16, 1]]
    c[17, :] = [kk[16, 0] - 40, kk[17, 1] - 20]
    c[18, :] = [kk[0, 0] + 40, kk[18, 1] - 20]

    return c
