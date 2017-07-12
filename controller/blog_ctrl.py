# coding:utf-8
'''
Created on 2017/4/12.

@author: Dxq
'''
import time
from common import base, tools
import config

mdb = config.mdb


class DailySign(base.BaseHandler):
    def post(self):
        ts = int(time.time())
        date = tools.ts2str(ts, '%Y-%m-%d')
        sign_rec = mdb.daily_sign.find_one({"date": date})
        if not sign_rec:
            daily_sign = {
                "_id": base.getRedisID("daily_sign"),
                "date": date,
                "date_time": ts
            }
            mdb.daily_sign.insert(daily_sign)
        return self.finish(base.rtjson())
