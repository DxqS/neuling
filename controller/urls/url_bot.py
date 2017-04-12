# coding:utf-8
'''
Created on 2017/4/12.

@author: Dxq
'''
from controller import bot_ctrl

urls = [
    ('/bot', bot_ctrl.Main),
    ('/bot/two', bot_ctrl.Main2)
]
