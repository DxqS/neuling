# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import time
import config
from common import tools


def ftime(timestamp, format='%Y-%m-%d %H:%M', short=False):
    timestamp = float(timestamp)
    if not short:
        return time.strftime(format, time.localtime(timestamp))

    diff = (time.time() - timestamp)
    if diff < (60 * 1):
        fdate = '刚刚'
    elif diff < (60 * 60 * 1):
        fdate = str(int(diff / 60)) + '分钟前'
    elif diff < (60 * 60 * 24 * 1):
        fdate = str(int(diff / (60 * 60))) + '小时前'
    elif diff < (60 * 60 * 24 * 7):
        fdate = str(int(diff / (60 * 60 * 24))) + '天前'
    else:
        fdate = time.strftime(format, time.localtime(timestamp))
    return fdate


def fdate(timestamp, format='%Y-%m-%d'):
    ts = tools.str2ts(tools.ts2str(time.time(), '%Y-%m-%d'), '%Y-%m-%d')
    if timestamp - ts > 0:
        return '今天'
    elif ts - timestamp < 24 * 60 * 60:
        return '昨天'
    else:
        return time.strftime(format, time.localtime(timestamp))


def mediaTime(times):
    return str(times / 60) + "分" + str(times % 60) + '秒'


def hidePhoen(phone):
    return phone[:3] + "*****" + phone[8:]


def LocalImg(path):
    return config.gconf['domain'] + path


def LabelName(label):
    keyval = {
        'TMKA': '甜美可爱', 'MLSS': '魅力时尚', 'QCJJ': '清纯简洁',
        'ZRYY': '自然优雅', 'GYRM': '高雅柔美', 'ZXCZ': '知性沉着',
        'LMMR': '浪漫迷人', 'HLGY': '华丽高雅', 'XDMD': '现代摩登'
    }
    return keyval[label]


filters = {
    'ftime': ftime,
    'fdate': fdate,
    'mediaTime': mediaTime,
    'LabelName': LabelName,
    'LocalImg': LocalImg
}
