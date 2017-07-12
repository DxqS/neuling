# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''
from controller import myself_ctrl

urls = [
    ('/my/base/info', myself_ctrl.BaseInfo),
    ('/my/daily/life', myself_ctrl.DailyLife),
    ('/my/her', myself_ctrl.HerLife)
]