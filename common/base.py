# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
from __future__ import absolute_import, division, print_function, with_statement

import os
import web
import time
import functools

from web.contrib.template import render_jinja
from tornado.web import RequestHandler

import config
from common import myfilter


import datetime
import numbers
import os.path
from tornado.util import (import_object, ObjectDict, raise_exc_info,
                          unicode_type, _websocket_mask, re_unescape, PY3)

if PY3:
    import urllib.parse as urlparse
    from urllib.parse import urlencode
else:
    import urlparse
    from urllib import urlencode

try:
    import typing  # noqa

    # The following types are accepted by RequestHandler.set_header
    # and related methods.
    _HeaderTypes = typing.Union[bytes, unicode_type,
                                numbers.Integral, datetime.datetime]
except ImportError:
    pass

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


def login_auth(method):
    """
    Dxq登入授权
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.get_secure_cookie('auth'):
            self.set_secure_cookie('auth', str(int(time.time())))
            if self.request.method in ("GET", "HEAD"):
                url = self.get_login_url()
                if "?" not in url:
                    if urlparse.urlsplit(url).scheme:
                        # if login url is absolute, make next absolute too
                        next_url = self.request.full_url()
                    else:
                        next_url = self.request.uri
                    url += "?" + urlencode(dict(next=next_url))
                self.redirect(url)
                return
            raise web.HTTPError(403)
        return method(self, *args, **kwargs)
    return wrapper
