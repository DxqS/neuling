# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import time
from common import base
from service import picture_service


# @base.login_auth
class Index(base.BaseHandler):
    def get(self):
        # auth_time = int(self.get_secure_cookie('auth'))
        # time_left = 86400 + auth_time - int(time.time())
        return self.render('dxq_picture/index.html')

    def post(self):
        back = picture_service.Base64_to_File(self.input("back"), 'merge')
        sub = picture_service.Base64_to_File(self.input("sub"), 'merge')
        picture = picture_service.Poster_Add(back, sub, (100, 100, 100))
        return self.finish(base.rtjson(picture=picture))


class Sketch(base.BaseHandler):
    def get(self):
        return self.render('dxq_picture/sketch.html')

    def post(self):
        back = picture_service.Base64_to_File(self.input("back"), 'merge')
        sub = picture_service.Base64_to_File(self.input("sub"), 'merge')
        picture = picture_service.Poster_Add(back, sub, (100, 100, 100))
        return self.finish(base.rtjson(picture=picture))
