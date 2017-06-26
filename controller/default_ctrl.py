# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import time
from common import base


class Login(base.BaseHandler):
    def get(self):
        return self.render('login/login.html')

    def post(self):
        print self.request.remote_ip

        print self.set_cookie(str(self.request.remote_ip).replace(":","").replace(":",""), str(int(time.time())))
        return self.finish(base.rtjson())
