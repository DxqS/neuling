# coding:utf-8
'''
Created on 2017/7/4.

@author: chk01
'''
import os
# import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')

import tornado.ioloop
import tornado.log
import tornado.web
import tornado.wsgi
import tornado.options

import config
import controller

settings = dict(
    blog_title=u"Tornado Blog",
    template_path=os.path.join(os.path.dirname(__file__), "views"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    # ui_modules={"Entry": EntryModule},
    # xsrf_cookies=True,
    cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
    login_url="/login",
    # debug=False if config.run_mode == 'release' else True
)


def make_app():
    tornado.options.options.log_to_stderr = True
    tornado.options.options.logging = 'warning'
    tornado.options.options.log_file_prefix = config.srv['log']['application.log']
    tornado.log.enable_pretty_logging(tornado.options.options)
    application = tornado.web.Application(controller.urls.ctrls, **settings)

    return application


dxqwsgi = tornado.wsgi.WSGIAdapter(make_app())
