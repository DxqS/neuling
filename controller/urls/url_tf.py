# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''
from controller import tf_ctrl

urls = [
    ('/tf/index', tf_ctrl.Index),
    ('/tf/add', tf_ctrl.Add),
    ('/tf/edit/(\d+)', tf_ctrl.Edit),
]
