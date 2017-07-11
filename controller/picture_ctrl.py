# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import time
from common import base


class Index(base.BaseHandler):
    @base.login_auth
    def get(self):
        auth_time = int(self.get_secure_cookie('auth'))
        time_left = 86400 + auth_time - int(time.time())
        return self.render('dxq_picture/material_list.html', time_left=time_left)
