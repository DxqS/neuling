# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
from common import base

class Login(base.BaseHandler):
    @base.login_auth
    def get(self):
        return self.render('login/login.html')

    def post(self):
        return self.finish(base.rtjson())
