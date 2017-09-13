# coding:utf-8
'''
Created on 2017/4/12.

@author: Dxq
'''

import tornado.ioloop
import tornado.web
import tornado.wsgi
import os
import controller

PORT = 8888
settings = dict(
    blog_title=u"Tornado Blog",
    template_path=os.path.join(os.path.dirname(__file__), "views"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    # ui_modules={"Entry": EntryModule},
    # xsrf_cookies=True,
    static_hash_cache=True,
    cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
    login_url="/login",
    # debug=True,
)

def make_app():
    return tornado.web.Application(controller.urls.ctrls, **settings)



# application = tornado.wsgi.WSGIAdapter(make_app())
if __name__ == "__main__":
    app = make_app()
    app.listen(PORT)
    print("http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
