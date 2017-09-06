# coding:utf-8
'''
Created on 2017/9/6.

@author: chk01
'''
import numpy as np
import struct
import scipy.io as scio


def loadImageSet(filename):
    '''
    :param filename:二进制文件名,MNIST数据集下载解压出的文件
    :return: images shape 为[60000,784]的数组
    '''
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置

    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    # Dxq Add
    # 将 数字转为float32 并且将范围[0,255]->[0,1]
    images = np.multiply(imgs.astype(np.float32), 1.0 / 255.0)
    return images, head


def loadLabelSet(filename):
    '''
    :param filename: 二进制文件名,MNIST数据集下载解压出的文件
    :return: labelArray shape 为[60000,10]的数组
    '''
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    labelNum = head[1]
    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labelList = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    # Dxq Add
    # 将[1,2,3,0...7,9...]的标签形式转成[0,1,0,0,0,0,0,0,0,0]的形式
    labelArray = np.zeros([labelNum, 10])
    for i, label in enumerate(labelList):
        labelArray[i][label] = 1

    return labelArray, head


#
if __name__ == "__main__":
    '''
    运行该脚本需要先下载MNIST并解压，将文件放置同级目录
    '''
    file1 = 'train-images.idx3-ubyte'
    file2 = 'train-labels.idx1-ubyte'

    images, data_head = loadImageSet(file1)
    labels, labels_head = loadLabelSet(file2)

    # 将数据保存到mat文件
    scio.savemat('../resource/mnist_data.mat', {'X': images, 'Y': labels})
    # 从mat文件读取数据
    # data = scio.loadmat('mnist_data.mat')
