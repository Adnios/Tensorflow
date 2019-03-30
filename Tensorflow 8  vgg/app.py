#coding:utf-8
import numpy as np
import tensorflow as tf
#引入绘图模块
import matplotlib.pyplot as plt
#引用自定义模块
import vgg16
import utils
from Nclasses import labels

testNum = input("input the number of test pictures:")
for i in range(testNum):
    img_path = raw_input('Input the path and image name:')
	#对待测试图像出预处理操作
    img_ready = utils.load_image(img_path) 

    #定义画图窗口，并指定窗口名称
    fig=plt.figure(u"Top-5 预测结果") 

    with tf.Session() as sess:
		#定义一个维度为[1, 224, 224, 3]的占位符
        images = tf.placeholder(tf.float32, [1, 224, 224, 3])
		#实例化出vgg
        vgg = vgg16.Vgg16() 
		#前向传播过程，调用成员函数，并传入待测试图像
        vgg.forward(images) 
		#将一个batch数据喂入网络，得到网络的预测输出
        probability = sess.run(vgg.prob, feed_dict={images:img_ready})
        #得到预测概率最大的五个索引值
		top5 = np.argsort(probability[0])[-1:-6:-1]
        print "top5:",top5
		#定义两个list-对应概率值和实际标签
        values = []
        bar_label = []
		#枚举上面取出的五个索引值
        for n, i in enumerate(top5): 
            print "n:",n
            print "i:",i
			#将索引值对应的预测概率值取出并放入value
            values.append(probability[0][i]) 
			#将索引值对应的际标签取出并放入bar_label
            bar_label.append(labels[i]) 
            print i, ":", labels[i], "----", utils.percent(probability[0][i]) 
        
		#将画布分为一行一列，并把下图放入其中
        ax = fig.add_subplot(111) 
		#绘制柱状图
        ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
        #设置横轴标签
		ax.set_ylabel(u'probabilityit') 
		#添加标题
        ax.set_title(u'Top-5') 
        for a,b in zip(range(len(values)), values):
			#显示预测概率值
            ax.text(a, b+0.0005, utils.percent(b), ha='center', va = 'bottom', fontsize=7)   
        #显示图像
		plt.show() 


    
