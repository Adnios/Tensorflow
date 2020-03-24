# coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resize_height = 28
resize_width = 28


# 生成tfrecords文件
def write_tfRecord(tfRecordName, image_path, label_path):
    # 新建一个writer
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    # 循环遍历每张图和标签
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1
        # 把每张图片和标签封装到example中
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        # 把example进行序列化
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture:", num_pic)
    # 关闭writer
    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


# 解析tfrecords文件
def read_tfRecord(tfRecord_path):
    # 该函数会生成一个先入先出的队列，文件阅读器会使用它来读取数据
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    # 新建一个reader
    reader = tf.TFRecordReader()
    # 把读出的每个样本保存在serialized_example中进行解序列化，标签和图片的键名应该和制作tfrecords的键名相同，其中标签给出几分类。
    _, serialized_example = reader.read(filename_queue)
    # 将tf.train.Example协议内存块(protocol buffer)解析为张量
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    # 将img_raw字符串转换为8位无符号整型
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 将形状变为一行784列
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    # 变成0到1之间的浮点数
    label = tf.cast(features['label'], tf.float32)
    # 返回图片和标签
    return img, label


def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    # 随机读取一个batch的数据
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=700)
    # 返回的图片和标签为随机抽取的batch_size组
    return img_batch, label_batch


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()