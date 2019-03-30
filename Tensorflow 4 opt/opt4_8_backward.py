#coding:utf-8
#tensorflow学习笔记(北京大学) tf4_8_backward.py 完全解析  
#QQ群：476842922（欢迎加群讨论学习）
#如有错误还望留言指正，谢谢
#0导入模块 ，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import opt4_8_generateds
import opt4_8_forward

STEPS = 40000
BATCH_SIZE = 30 
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01#正则化

def backward():#反向传播
	x = tf.placeholder(tf.float32, shape=(None, 2))#占位
	y_ = tf.placeholder(tf.float32, shape=(None, 1))#占位
	#X：300行2列的矩阵。Y_:坐标的平方和小于2，给Y赋值1，其余赋值0
	X, Y_, Y_c = opt4_8_generateds.generateds()
	
	y = opt4_8_forward.forward(x, REGULARIZER)#前向传播计算后求得输出y
	
	global_step = tf.Variable(0,trainable=False)#轮数计数器	
	#指数衰减学习率
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,#学习率
		global_step,#计数
		300/BATCH_SIZE,
		LEARNING_RATE_DECAY,#学习衰减lü
		staircase=True)#选择不同的衰减方式


	#定义损失函数
	loss_mse = tf.reduce_mean(tf.square(y-y_))#均方误差
	loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))#正则化
	
	#定义反向传播方法：包含正则化
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()#初始化
		sess.run(init_op)#初始化
		for i in range(STEPS):
			start = (i*BATCH_SIZE) % 300
			end = start + BATCH_SIZE#3000轮
			sess.run(train_step, feed_dict={x: X[start:end], y_:Y_[start:end]})
			if i % 2000 == 0:
				loss_v = sess.run(loss_total, feed_dict={x:X,y_:Y_})
				print("After %d steps, loss is: %f" %(i, loss_v))

		xx, yy = np.mgrid[-3:3:.01, -3:3:.01]#网格坐标点
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = sess.run(y, feed_dict={x:grid})
		probs = probs.reshape(xx.shape)
	
	plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c)) #画点
	plt.contour(xx, yy, probs, levels=[.5])#画线
	plt.show()#显示图像
	
if __name__=='__main__':
	backward()
