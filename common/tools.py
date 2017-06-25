# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
# coding:utf-8

import time
import string
import random
import urllib
import requests

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def int2hex(num):
    return str(hex(int(num))).replace('0x', '').replace('L', '')


def hex2int(shex):
    return int('0x' + shex, 16)


def str2intList(strList):
    intList = [int(s) for s in strList]
    return intList


def hex2intList(hexList):
    intList = [hex2int(shex) for shex in hexList]
    return intList


def str2ts(dts, format='%Y-%m-%d %H:%M'):
    return time.mktime(time.strptime(dts, format))


def ts2str(ts, format='%Y-%m-%d %H:%M'):
    return time.strftime(format, time.localtime(ts))


def sxml2dict(sxml):
    redict = {}
    root = ET.fromstring(sxml)
    for child in root:
        value = child.text
        redict[child.tag] = value
    return redict


def dict2sxml(obj):
    xml = ["<xml>"]
    for k, v in obj.iteritems():
        if v.isdigit():
            xml.append("<{0}>{1}</{0}>".format(k, v))
        else:
            xml.append("<{0}><![CDATA[{1}]]></{0}>".format(k, v))
    xml.append("</xml>")
    return "".join(xml)


def group(lst, block):
    return [lst[i:i + block] for i in range(0, len(lst), block)]


def uniqueName():
    s1 = int2hex(time.time() * 1000)
    s2 = int2hex(random.randint(100000, 999999))
    return s2 + s1


def randomList(num, docs):
    rid = []
    for i in range(len(docs)):
        r = random.randrange(0, len(docs))
        if r not in rid:
            rid.append(r)
    return [docs[j] for j in rid[:num]]


def createNoncestr(length=16, rule=string.ascii_letters + string.digits):
    return ''.join(random.sample(rule, length))


def createLinkString(parameter):
    parameter = parameter.items()
    parameter.sort()
    return '&'.join(["%s=%s" % (k, v) for k, v in parameter])


def createLinkstringUrlencode(parameter):
    parameter = parameter.items()
    parameter.sort()
    kv_list = [(k, urllib.quote(v.encode('utf-8') if type(v).__name__ == "str" else v)) for k, v in parameter]
    return '&'.join(["%s=%s" % (k, str(v)) for k, v in kv_list])


def httpGet(url, param):
    resp = requests.get(url, param)
    return resp.json()


def httpPost(url, param):
    resp = requests.post(url, param)
    return resp.json()


def query2list(query, mode='query'):
    """
    sql query对象转list
    """
    return [i.to_dict() for i in query] if mode == "query" else [i for i in query]
