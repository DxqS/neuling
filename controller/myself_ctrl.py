# coding:utf-8
'''
Created on 2017/7/13.

@author: Dxq
'''
from common import base
from service import picture_service
from config import mdb


class BaseInfo(base.BaseHandler):

    def get(self):
        return self.render('login/base_info.html')

    def post(self):
        name = self.input("name")
        nickname = self.input("nickname")
        mobile = self.input("mobile")
        avatar = picture_service.Base64_to_File(self.input("avatar"), 'avatar')
        user = {
            '_id': base.getRedisID('user'),
            'name': name,
            'nickname': nickname,
            'mobile': mobile,
            'avatar': avatar
        }
        mdb.user.insert_one(user)
        return self.finish(base.rtjson())


class DailyLife(base.BaseHandler):
    def get(self):
        return self.render('login/lw-article-fullwidth.html')


class HerLife(base.BaseHandler):
    def get(self):
        return self.render('login/lw-category-sidebar.html')
