# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''
from controller import tf_ctrl

urls = [
    ('/tf/source/index', tf_ctrl.SourceIndex),
    ('/tf/source/add', tf_ctrl.SourceAdd),
    ('/tf/source/edit/(\d+)', tf_ctrl.SourceEdit),

    ('/tf/user/index', tf_ctrl.UserIndex),
    ('/tf/user/add', tf_ctrl.UserAdd),

    ('/tf/train/index', tf_ctrl.TrainIndex),
    ('/tf/train/test', tf_ctrl.TrainTest),

]
