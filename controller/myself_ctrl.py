# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''
from common import base


class BaseInfo(base.BaseHandler):
    def get(self):
        return self.render('login/lw-article.html')


class DailyLife(base.BaseHandler):
    def get(self):
        return self.render('login/lw-article-fullwidth.html')


class HerLife(base.BaseHandler):
    def get(self):
        return self.render('login/lw-category-sidebar.html')
