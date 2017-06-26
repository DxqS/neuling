# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
from common import base


class Index(base.BaseHandler):
    def get(self):
        print self.get_cookie(str(self.request.remote_ip))
        return self.render('dxq_picture/merge.html')
