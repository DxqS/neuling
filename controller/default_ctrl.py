# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
from common import base


class Login(base.BaseHandler):
    def get(self):
        return self.render('login/login.html')
