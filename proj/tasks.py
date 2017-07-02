# coding:utf-8
'''
Created on 2017/7/2.

@author: Dxq
'''
from __future__ import absolute_import
from common import base
import requests

import config
from proj.celery import app

HOUR = 3600
DAY = 24 * HOUR
mdb = config.mdb


@app.task
def add(x, y):
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def url_call(url, **args):
    r = requests.post(url, data=args)
    return r.text


@app.task
def tt():
    mdb.user.insert({"_id": base.getRedisID("user"), "name": "ss"})
    return True
