# coding:utf-8
'''
Created on 2017/6/26.

@author: Dxq
'''
# from proj import tasks
# #
# tasks.tt.delay()
# print 12


#
# mdb = config.mdb
# user = {
#     '_id': base.getRedisID('user'),
#     'name': "莫笑刀3",
#     'date':  datetime.datetime.today()
# }
# mdb.user.insert(user)
from service.picture_service import cutSqure
from PIL import Image
img = Image.open('3.jpg')
img = cutSqure(img)
img.save('4.jpg')
