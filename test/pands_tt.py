# coding:utf-8
'''
Created on 2017/9/16.

@author: chk01
'''
import pandas as pd
import numpy as np

tt = pd.Series([1, 23, 3], ['a', 'c', 'd'])
# print(tt)
# print(pd.DataFrame({'res': [tt.median(), tt.mean(), tt.std()]}, index=['mid', 'mean', 'std']))
ss = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
      'year': [2000, 2001, 2002, 2001, 2002],
      'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
sss = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
sss2 = np.array([[1, 2, 3, 3, 4], [4, 2, 4, 5, 5]])
sssss = [sss, sss2]
# frame = pd.DataFrame({"sss":sssss})
# frame.to_csv('1.train')
# print(frame['sss'])
# print(np.array(frame['sss']))
# print(np.array(frame['sss'])[0])
# print(np.array(frame['sss'])[0].shape)
df_train = pd.read_csv("1.train", skipinitialspace=True)
print(df_train.values)
print(df_train.values[0])
print(type(df_train.values[0]))
print(np.array(df_train.values[1][1]))

# print(np.array(df_train['sss'])[0])
# print(np.array(np.array(df_train['sss'])[0]))
# print(np.array(np.array(df_train['sss'])[0]))
# print(type(np.array(np.array(df_train['sss'])[0])))
# dates = pd.date_range('20131228', periods=6)
# print(dates)
