# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import time

from common import base


class Login(base.BaseHandler):
    @base.login_auth
    def get(self):
        return self.render('login/login.html')

    def post(self):
        self.set_secure_cookie('auth', str(int(time.time())))
        return self.finish(base.rtjson())


class Index(base.BaseHandler):
    def get(self):
        return self.render('login/index.html')

    def post(self):
        self.set_secure_cookie('auth', str(int(time.time())))
        return self.finish(base.rtjson())
