# coding:utf-8
'''
Created on 2017/7/5.

@author: chk01
'''
import config
import re
from common import base
import pymongo

mdb = config.mdb
# 增
# user = {
#     '_id': base.getRedisID('user'),
#     'name': "莫笑刀3"
# }
# mdb.user.insert(user)
# print mdb.user.insert_one(user)
# 记数
# print mdb.user.count()

# 查询
# print mdb.user.find_one({"_id": 100001})
# mdb.user.find_one({"name": re.compile('^刀')})

# 常用查询条件
# 比较查询
# '$eq', 'gt', 'lt', 'gte', 'ne', 'in', 'nin'
# 逻辑查询
# '$or','and','not','nor'
# mdb.user.find({"$or": [{"_id": 100001}, {"name": "莫笑刀2"}]})
# mdb.user.find({"$and": [{"$or": [{"_id": 100001}, {"name": "莫笑刀2"}]}, {"_id": 100001}]})
# mdb.user.find({"name": {"$not": re.compile('^刀')}})
# '$exists', 'type'
# mdb.user.find({"age": {"$type": 4}})
# mdb.user.find({"age": {"$type": "array"}})
# 4:array
# 计算查询
# '$mod', 'regex', 'text', 'where'
# mdb.user.find({"$where": "function(){return obj.age == obj.ega}"})

# 结构固定
# places = {
#     "_id": base.getRedisID('places'),
#     "location": {
#         "type": "Point",
#         "coordinates": [-73.97, 40.77]
#     },
#     "name": "Central Park",
#     "category": "Parks"
# }
# mdb.places.insert(places)
# mdb.places.create_index([("location", "2dsphere")])
# mdb.places.find({"location": {
#     "$nearSphere": {"$geometry": {"type": "Point", "coordinates": [-73.9, 40.7]}, "$maxDistance": 1,
#                     "$minDistance": 1}}})
# mdb.places.find({"location": {"$geoWithin": {"$center": [[-73, 40], 10000]}}})

# '$geometry','center'
# mdb.places.find({"location": {"$geoWithin": {"$box": [[-100, 0], [100, 100]]}}})
# mdb.places.find({"location": {"$geoWithin": {"$polygon": [[-100, 0], [100, 100], [-100, 100]]}}})

# '$all', 'elemMatch', 'size'
# mdb.places.find({"category": {"$all": ["Parks", "Pa"]}})
# mdb.user.find({"age": {"$elemMatch": {"$gte": 80, "$lt": 85}}})
# mdb.user.find({"age": {"$gte": 85}}, {"age.$": 1})
# mdb.user.find({}, {"age": {"$elemMatch": {"index": {"$gt": 1}}}})
# mdb.user.find({}, {"age": {"$elemMatch": {"index": {"$gt": 1}}}})
# mdb.user.find({}, {"age": {"$slice": [0, 1]}})
# mdb.user.update_one({"_id": 100001}, {"$inc": {"age.0": 1, "ega": 1}})
# mdb.user.update_one({"_id": 100001}, {"$mul": {"price": 21}})
# mdb.user.update({"_id": 100001}, {"$rename": {'ega': 'age2'}})
# mdb.user.update({"_id": 100001}, {"$rename": {'age2.first': 'age2.fname'}})

# "$addToSet", "pop", 'pullAll', 'pull', 'push', 'each', 'slice', 'sort', 'position'
# 优先执行update 若查询为无执行upsert->响应setOnInsert 内容
# mdb.user.update({"name": "莫笑刀"}, {"$set": {"age": 112},
#                                   "$setOnInsert": {"_id": base.getRedisID("user")}}, upsert=True)
# 删除字段
# mdb.user.update({"_id": 100007}, {"$unset": {"item": "", "instock": ""}})
# mdb.user.update({"_id": 100007, "grades.grade": 85}, {"$set": {"grades.$.std": 6}})
# 删除所有scour（Array）中的0，5
# mdb.user.update({"_id": 100007}, {"$pullAll": {"score": [0, 5]}})

# mdb.user.update({"_id": 100007}, {"$push": {"quizzes": {
#     "$each": [{"wk": 2, "score": 88}, {"wk": 2, "score": 99}],
#     "$sort": {"score": -1},
#     "$slice": 3
# }}})

# mdb.user.update({"_id": 100007}, {"$push": {"scores": {"$each": [20, 30], "$position": 1}}})

# 'collStats', 'project', 'match', 'redact', 'limit', 'skip', 'unwind', 'group', 'sample', 'sort', 'geoNear', 'lookup'
# 'out', 'indexStats', 'facet', 'bucket', 'bucketAuto', 'sortByCount', 'addFields', 'replaceRoot'
# 'count', 'graphLookup'
# project = {'$project': {'temp_mod': 1, 'ctime': 1, 'uid': 1}}
# mdb.user.aggregate([project])
# project = {
#     "$project": {
#         "title": 1,
#         "isbn": {
#             "prefix": {"$substr": ["$isbn", 0, 3]},
#             "group": {"$substr": ["$isbn", 3, 2]},
#             "publisher": {"$substr": ["$isbn", 5, 4]},
#             "title": {"$substr": ["$isbn", 9, 3]},
#         },
#         "lastName": "$author.last",
#         "copiesSold": "$copies"
#     }
# }
# mdb.user.aggregate([project])
# 性能
# mdb.user.find({}).explain()
