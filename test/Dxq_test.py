# coding:utf-8
'''
Created on 2017/6/26.

@author: Dxq
'''

import config
import numpy as np

rdb = config.rdb
rdb.lpush('tt', np.array([2, 3, 3]))
ss = [np.array(list(t)) for t in rdb.lrange('tt', 0, -1)]
print(ss)
for i in rdb.lrange('tt', 0, -1):
    print(i)
    print(type(i))
