# coding:utf-8
'''
Created on 2017/9/16.

@author: chk01
'''
import pandas as pd

tt = pd.Series([1, 23, 3], ['a', 'c', 'd'])
print(tt)
print(pd.DataFrame({'res': [tt.median(), tt.mean(), tt.std()]}, index=['mid', 'mean', 'std']))
ss = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
      'year': [2000, 2001, 2002, 2001, 2002],
      'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(ss)
print(frame)
print(frame['pop'])

dates = pd.date_range('20131228', periods=6)
print(dates)
