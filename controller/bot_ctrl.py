# coding:utf-8
'''
Created on 2017/4/12.

@author: Dxq
'''
import tornado.web



class Main(tornado.web.RequestHandler):
    def get(self):
        self.render("bot/dxq.html")


class Main2(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world2")
