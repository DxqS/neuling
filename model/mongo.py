# coding:utf-8
'''
Created on 2017/9/7.

@author: chk01
'''
style_source = {
    "_id": 100004,  # ID

    "outline": "曲",
    "sense": "小",
    "path": "/resource/style/origin/TMKA/100004.jpg",
    "type": "train",  # 集合类型 ['train','test']
    "result": {
        "right_eye": "/resource/style/result/right_eye/TMKA/100004.jpg",
        "right_eyebrow": "/resource/style/result/right_eyebrow/TMKA/100004.jpg",
        "chin": "/resource/style/result/chin/TMKA/100004.jpg",
        "nose_bridge": "/resource/style/result/nose_bridge/TMKA/100004.jpg",
        "top_lip": "/resource/style/result/top_lip/TMKA/100004.jpg",
        "left_eye": "/resource/style/result/left_eye/TMKA/100004.jpg",
        "bottom_lip": "/resource/style/result/bottom_lip/TMKA/100004.jpg",
        "left_eyebrow": "/resource/style/result/left_eyebrow/TMKA/100004.jpg",
        "nose_tip": "/resource/style/result/nose_tip/TMKA/100004.jpg"
    },
    "label": "TMKA",  # 标签['TMKA', 'MLSS', 'QCJJ','ZRYY', 'GYRM', 'ZXCZ','LMMR', 'HLGY', 'XDMD']
    # 五官数据
    # 轮廓
    "chin": [[385, 470], [391, 512], [400, 553], [409, 593], [421, 632], [443, 664],
             [472, 689], [509, 706], [551, 711], [592, 705], [629, 686], [659, 659],
             [681, 627], [693, 589], [703, 551], [711, 511], [717, 469]],

    # 鼻子
    "nose_bridge": [[551, 472], [551, 503], [551, 531], [551, 561]],
    # 鼻子
    "nose_tip": [[518, 576], [534, 581], [550, 585], [567, 580], [582, 574]],

    # 上嘴唇
    "top_lip": [[492, 618], [516, 610], [536, 605],
                [550, 609], [565, 605], [585, 610],
                [609, 615], [599, 617], [565, 620],
                [550, 621], [536, 620], [501, 619]],
    # 下嘴唇
    "bottom_lip": [[609, 615], [585, 631], [565, 639],
                   [550, 640], [535, 639], [515, 632],
                   [492, 618], [501, 619], [536, 621],
                   [550, 622], [565, 620], [599, 617]],
    # 左眼
    "left_eye": [[448, 478], [467, 467], [487, 468],
                 [502, 482], [485, 483], [465, 483]],
    # 右眼
    "right_eye": [[597, 481], [614, 467], [634, 466],
                  [651, 476], [635, 482], [615, 482]],

    # 右眉
    "right_eyebrow": [[583, 437], [610, 428], [637, 426], [664, 430], [685, 445]],
    # 左眉
    "left_eyebrow": [[414, 444], [435, 427], [463, 422], [492, 425], [518, 436]]
}
