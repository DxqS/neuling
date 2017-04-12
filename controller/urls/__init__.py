# coding:utf-8
'''
Created on 2017/4/12.

@author: Dxq
'''
import os

ctrls = []
for f in os.listdir(os.path.split(__file__)[0]):
    module_name, ext = os.path.splitext(f)
    if module_name.startswith('url_') and ext == '.py':
        module = __import__(__name__ + '.' + module_name, fromlist=module_name)
        for i in module.urls:
            ctrls.append(i)
