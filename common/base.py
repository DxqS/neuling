# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import os
import web
import time

from web.contrib.template import render_jinja
from tornado.web import RequestHandler

import config
from common import myfilter
errorDesc = config.errorDesc

def rtjson(code=1, **args):
    """return json"""
    if code == 1:
        args['status'] = 1
    else:
        args['status'] = 0
        args['error_code'] = code
        args['error_msg'] = errorDesc.get(code)
    return args


def render(module_path, prefix=''):
    gconf = config.gconf
    gconf['debug'] = web.config.debug
    gconf['uptime'] = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    tempath = prefix + os.path.split(module_path)[1].replace('.pyc', '').replace('.py', '').replace('_ctrl', '')

    render = render_jinja(['views/', 'views/' + tempath], encoding='utf-8')
    # render._lookup.globals.update(session=web.config._session, gconf=gconf)
    render._lookup.globals.update(gconf=gconf)
    render._lookup.filters.update(myfilter.filters)

    return render


class BaseHandler(RequestHandler):
    def get_current_user(self):
        return self.get_secure_cookie("uid")

    def input(self, name, default=None, strip=True):
        return self._get_argument(name, default, self.request.arguments, strip)

    def get_template_namespace(self):
        namespace = super(BaseHandler, self).get_template_namespace()
        namespace.update(myfilter.filters)
        namespace['gconf'] = config.gconf
        namespace['session'] = {'uname': self.get_current_user()}

        return namespace
