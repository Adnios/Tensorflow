#!/usr/bin/python
#coding:utf-8
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif']=['SimHei'] # 正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False # 正常显示正负号

def load_image(path):
    fig = plt.figure("Centre and Resize")
	#传入读入图片的参数路径
    img = io.imread(path) 
	#将像素归一化处理到[0,1]
    img = img / 255.0 
    
	#将该画布分为一行三列，把下面的图像放在画布的第一个位置
    ax0 = fig.add_subplot(131)  
	#添加子标签
    ax0.set_xlabel(u'Original Picture') 
	#添加展示该图像
    ax0.imshow(img) 
    
	#找到该图像的最短边
    short_edge = min(img.shape[:2]) 
	#把图像的w和h分别减去最短边，并求平均
    y = (img.shape[0] - short_edge) / 2  
    x = (img.shape[1] - short_edge) / 2 
	#取出切分过的中心图像
    crop_img = img[y:y+short_edge, x:x+short_edge] 
    
	#把下面的图像放在画布的第二个位置
    ax1 = fig.add_subplot(132) 
	#添加子标签
    ax1.set_xlabel(u"Centre Picture") 
	#添加展示该图像
    ax1.imshow(crop_img)
    
	#resize成固定的imagesize
    re_img = transform.resize(crop_img, (224, 224)) 
    
	#把下面的图像放在画布的第三个位置
    ax2 = fig.add_subplot(133) 
    ax2.set_xlabel(u"Resize Picture") 
    ax2.imshow(re_img)
	#转换为需要的输入形状
    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready

#定义百分比转换函数
def percent(value):
    return '%.2f%%' % (value * 100)

