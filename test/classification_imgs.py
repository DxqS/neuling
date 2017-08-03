# coding:utf-8
'''
Created on 2017/8/3.

@author: Dxq
'''
import os
import shutil
import xlrd

LabelDict = {
    'GYRM': [], 'HLGY': [], 'LMMR': [],
    'MLSS': [], 'QCJJ': [], 'TMKA': [],
    'XDMD': [], 'ZRYY': [], 'ZXCZ': []
}
en_ch = {
    'TMKA': '甜美可爱', 'MLSS': '魅力时尚', 'QCJJ': '清纯简洁',
    'ZRYY': '自然优雅', 'GYRM': '高雅柔美', 'ZXCZ': '知性沉着',
    'LMMR': '浪漫迷人', 'HLGY': '华丽高雅', 'XDMD': '现代摩登'
}
root_dir = 'C:\Users\Dxq\Desktop/aface_img'


def eachFile(Label):
    pathDir = os.listdir(root_dir + '/source')
    count = 0
    for sourceDir in pathDir:
        num = sourceDir.replace('ͼ', '').replace('.jpg', '')
        try:
            if num in LabelDict[Label]:
                count += 1
                sourceDir = root_dir + '/source/' + sourceDir
                targetDir = root_dir + '/face/' + Label
                shutil.copy(sourceDir, targetDir)
                print 'ok2'
        except ValueError:
            print num
    print count


def get_label_ids():
    data = xlrd.open_workbook(root_dir + '/result.xlsx')
    table = data.sheets()[0]
    # nrows = table.nrows
    # ncols = table.ncols
    for i in range(table.nrows):
        if i == 0:
            continue
        for key in LabelDict.keys():
            if table.row_values(i)[3].encode('utf-8') == en_ch[key]:
                # table.col_values(i)
                label_id = table.row_values(i)[0].encode('utf-8').replace("图", "")
                LabelDict[key].append(label_id)
    # print LabelDict.keys()
    for key in LabelDict.keys():
        eachFile(key)


# get_label_ids()
