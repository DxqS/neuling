# coding:utf-8
'''
Created on 2017/9/8.

@author: chk01
'''
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
root_dir = 'C:/Users/chk01/Desktop/style'


def eachFile(Label):
    pathDir = os.listdir(root_dir + '/source')
    for sourceDir in pathDir:
        num = sourceDir.replace('图', '').replace('.jpg', '')
        try:
            if num in LabelDict[Label]:
                sourceDir = root_dir + '/source/' + sourceDir
                targetDir = root_dir + '/' + Label
                if not os.path.exists(targetDir):
                    os.mkdir(targetDir)
                shutil.copy(sourceDir, targetDir)
        except ValueError:
            print(num)


def get_label_ids():
    data = xlrd.open_workbook(root_dir + '/result.xlsx')
    table = data.sheets()[0]
    # nrows = table.nrows
    # ncols = table.ncols
    for i in range(table.nrows):
        if i == 0:
            continue
        for key in LabelDict.keys():
            if table.row_values(i)[3] == en_ch[key]:
                # table.col_values(i)
                label_id = table.row_values(i)[0].replace("图", "")
                LabelDict[key].append(label_id)
    # print LabelDict.keys()
    for key in LabelDict.keys():
        eachFile(key)


if __name__ == "__main__":
    '''
    前提：此脚本需要将素材和素材分类表放到root_dir目录下
    功能：将素材根据分类表分到各自的文件夹中
    '''
    get_label_ids()
