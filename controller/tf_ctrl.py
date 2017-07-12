# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''
from common import base


class Index(base.BaseHandler):
    def get(self):
        return self.render('login/lw-img.html')
