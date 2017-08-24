# coding:utf-8
'''
Created on 2016/9/1

@author: Dxq
'''
import redis

args = {
    'host': 'dev.yanzijia.cn',
    'port': 6379,
    'db': 5,
    'password': 'yzjpwd123'
}

pool = redis.ConnectionPool(**args)
rdb = redis.StrictRedis(connection_pool=pool)

# Keys:指令测试
# rdb.set('k1', 1)
# rdb.set('k2', 2)
# rdb.set('k3', 3)
# print rdb.exists('k1')
# print rdb.exists('1k')
# print rdb.ttl('k1')
# print rdb.expire('k2', 20)
# print rdb.ttl('k2')
# print rdb.type('k1')

# String:指令测试
# rdb.set('k1', 1)
# rdb.set('k2', 2)
# rdb.set('k3', 3)

# del指令
# rdb.delete('k1')
#
#
# rdb.append('k1', 23)
# rdb.strlen('k1')
#
# rdb.incr('k1')
# rdb.incrby('k2', 10)
# rdb.getrange('k1', 1, 2)
# rdb.setrange('k2', 0, 7)
#
# rdb.setex('k4', 20, 4)
# rdb.setnx('k4', 20)
# rdb.setnx('k5', 20)
# print rdb.mget('k1', 'k2')
# rdb.mset({'k6': 6, 'k7': 7})
# print rdb.mget('k6', 'k7')
# rdb.msetnx({'k10': 77, 'k8': 8})
# print rdb.getset('k1', 2)
# print rdb.get('k1')

# List指令测试
# rdb.lpush('k1', 1)
# rdb.rpush('k1', 2)
# print rdb.lrange('k1', 0, -1)
# rdb.lpop('k1')
# print rdb.lindex('k1', 3)
# print rdb.llen('k1')
# rdb.lrem('k1', 1, 3)
# rdb.ltrim('k1', 0, 2)
# print rdb.lrange('k1', 0, -1)
# rdb.ltrim('k1', 0, 2)
# print rdb.lrange('k1', 0, -1)
# rdb.lpush('k2', 1, 2, 3, 4, 5)
# print rdb.lrange('k1', 0, -1)
# print rdb.lrange('k2', 0, -1)
# rdb.rpoplpush('k1', 'k2')
# print rdb.lrange('k1', 0, -1)
# print rdb.lrange('k2', 0, -1)
# rdb.lset('k1', 0, 9)
# 在k1链表的x前面添加java
# rdb.linsert('k1', 'before', 'x', 'java')

# Set指令
# rdb.sadd('s2', 1, 2, 6)
# print rdb.smembers('s1')
# print rdb.scard('s1')
# 类似exists 如果在 返回true
# print rdb.sismember('s1', 21)
# print rdb.sdiff('s1', 's2')
# print rdb.sinter('s1', 's2')
# print rdb.sunion('s1', 's2')
# rdb.zadd('z1', 1, 'zz1', 2, 'zz2')
# print rdb.zrange('z1', 0, -1, withscores=True)
# print rdb.zrangebyscore('z1', 0, 1, withscores=True)
