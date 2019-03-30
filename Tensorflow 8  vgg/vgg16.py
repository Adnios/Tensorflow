#!/usr/bin/python
#coding:utf-8

import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

#样本RGB的平均值
VGG_MEAN = [103.939, 116.779, 123.68] 

class Vgg16():
    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
			#返回当前工作目录
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy") 
			#遍历其内键值对，导入模型参数
            self.data_dict = np.load(vgg16_path, encoding='latin1').item() 

    def forward(self, images):
        
        print("build model started")
		#获取前向传播开始时间
        start_time = time.time() 
		#逐个像素乘以255
        rgb_scaled = images * 255.0 
		#从GRB转换彩色通道到BRG
        red, green, blue = tf.split(rgb_scaled,3,3) 
		#减去每个通道的像素平均值，这种操作可以移除图像的平均亮度值
		#该方法常用在灰度图像上
        bgr = tf.concat([     
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]],3)
        #构建VGG的16层网络（包含5段卷积，3层全连接），并逐层根据命名空间读取网络参数
		#第一段卷积，含有两个卷积层，后面接最大池化层，用来缩小图片尺寸
        self.conv1_1 = self.conv_layer(bgr, "conv1_1") 
		#传入命名空间的name，来获取该层的卷积核和偏置，并做卷积运算，最后返回经过激活函数后的值
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
		#根据传入的pooling名字对该层做相应的池化操作
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")
        
		#第二段卷积，包含两个卷积层，一个最大池化层
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

        #第三段卷积，包含三个卷积层，一个最大池化层
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")
        
		#第四段卷积，包含三个卷积层，一个最大池化层
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")
        
		#第五段卷积，包含三个卷积层，一个最大池化层
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")
        
		#第六层全连接
		#根据命名空间name做加权求和运算
        self.fc6 = self.fc_layer(self.pool5, "fc6")
		#经过relu激活函数
        self.relu6 = tf.nn.relu(self.fc6) 
        
		#第七层全连接
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        
		#第八层全连接
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")
        
		#得到全向传播时间
        end_time = time.time() 
        print(("time consuming: %f" % (end_time-start_time)))
        
		#清空本次读取到的模型参数字典
        self.data_dict = None 
    
	#定义卷积运算    
    def conv_layer(self, x, name):
		#根据命名空间name找到对应卷积层的网络参数
        with tf.variable_scope(name): 
			#读到该层的卷积核
            w = self.get_conv_filter(name) 
			#卷积运算
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME') 
            #读到偏置项
			conv_biases = self.get_bias(name) 
			#加上偏置，并做激活计算
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases)) 
            return result
    
	#定义获取卷积核的参数
    def get_conv_filter(self, name):
		#根据命名空间从参数字典中获取对应的卷积核
        return tf.constant(self.data_dict[name][0], name="filter") 
    
	#定义获取偏置项的参数
    def get_bias(self, name):
		#根据命名空间从参数字典中获取对应的偏置项
        return tf.constant(self.data_dict[name][1], name="biases")
    
	#定义最大池化操作
    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
	#定义全连接层的全向传播操作
    def fc_layer(self, x, name):
		#根据命名空间name做全连接层的计算
        with tf.variable_scope(name): 
			#获取该层的维度信息列表
            shape = x.get_shape().as_list() 
            dim = 1
            for i in shape[1:]:
				#将每层的维度相乘
                dim *= i 
			#改变特征图的形状，也就是将得到的多维特征做拉伸操作，只在进入第六层全连接层做该操作
            x = tf.reshape(x, [-1, dim])
			#读到权重值
            w = self.get_fc_weight(name) 
			#读到偏置项值
            b = self.get_bias(name) 
            #对该层输入做加权求和，再加上偏置
            result = tf.nn.bias_add(tf.matmul(x, w), b) 
            return result
    
	#定义获取权重的函数
    def get_fc_weight(self, name): 
		#根据命名空间name从参数字典中获取对应1的权重
        return tf.constant(self.data_dict[name][0], name="weights")

