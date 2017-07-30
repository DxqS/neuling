import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def position(im):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    ss = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    face_outline = ss[0:17, :]
    right_blow = ss[17:22, :]
    left_blow = ss[22:27, :]
    right_eye = ss[36:42, :]
    left_eye = ss[42:48, :]
    mouse = ss[48:66, :]
    nose = ss[27:36, :]
    return face_outline, right_blow, left_blow, right_eye, left_eye, mouse, nose, ss
