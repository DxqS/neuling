# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import time
from common import base
from service import picture_service as ps


# @base.login_auth
class Index(base.BaseHandler):
    def get(self):
        # auth_time = int(self.get_secure_cookie('auth'))
        # time_left = 86400 + auth_time - int(time.time())
        return self.render('dxq_picture/index.html')

    def post(self):
        back = self.input("back")
        sub = self.input("sub")
        if back and sub:
            back = ps.Base64_to_File(back, 'merge')
            subImg = ps.Base64_to_File(sub, 'merge')
        print back
        print subImg
        return self.finish(base.rtjson())
