# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
from common import base


class Index(base.BaseHandler):
    @base.login_auth
    def get(self):
        print self.get_secure_cookie('auth')
        return self.render('dxq_picture/merge.html')
