# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
from common import base


class Index(base.BaseHandler):
    def get(self):
        return self.render('picture/merge.html')
