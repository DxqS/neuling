# coding:utf-8
'''
Created on 2017/9/6.

@author: chk01
'''
import tensorflow as tf
import os
import tempfile
from six.moves import urllib
import numpy
import gzip


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(work_directory):
        tf.gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not tf.gfile.Exists(filepath):
        with tempfile.NamedTemporaryFile() as tmpfile:
            temp_file_name = tmpfile.name
            urllib.request.urlretrieve(SOURCE_URL + filename, temp_file_name)
            tf.gfile.Copy(temp_file_name, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.Size()
            print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

if __name__ == '__main__':
    if not tf.gfile.Exists('data'):
        tf.gfile.MakeDirs('data')
    filepath = os.path.join('data', TRAIN_IMAGES)

    train_images = extract_images(filepath)
    print(train_images)
