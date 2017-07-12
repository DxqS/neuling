# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import time
import config
from common import base, tools

mdb = config.mdb


class Login(base.BaseHandler):
    @base.login_auth
    def get(self):
        return self.render('login/login.html')

    def post(self):
        self.set_secure_cookie('auth', str(int(time.time())))
        return self.finish(base.rtjson())


class Index(base.BaseHandler):
    def get(self):
        date = tools.ts2str(time.time(), '%Y-%m-%d')
        sign_rec = mdb.daily_sign.find_one({"date": date})
        sign = 1 if sign_rec else 0
        return self.render('login/index.html', sign=sign)

    def post(self):
        self.set_secure_cookie('auth', str(int(time.time())))
        return self.finish(base.rtjson())
