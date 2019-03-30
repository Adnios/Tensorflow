#coding:utf-8
#tensorflow学习笔记(北京大学) tf4_8_forward.py 完全解析  
#QQ群：476842922（欢迎加群讨论学习）
#如有错误还望留言指正，谢谢
#前向传播就是搭建网络
import tensorflow as tf

#定义神经网络的输入、参数和输出，定义前向传播过程 
def get_weight(shape, regularizer):#（shape：W的形状，regularizer正则化）
	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)#赋初值，正态分布权重
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))#把正则化损失加到总损失losses中
	return w#返回w
#tf.add_to_collection(‘list_name’, element)：将元素element添加到列表list_name中

def get_bias(shape):  #偏执b
    b = tf.Variable(tf.constant(0.01, shape=shape)) #偏执b=0.01
    return b#返回值
	
def forward(x, regularizer):#前向传播
	
	w1 = get_weight([2,11], regularizer)#设置权重	
	b1 = get_bias([11])#设置偏执
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)#计算图

	w2 = get_weight([11,1], regularizer)#设置权重
	b2 = get_bias([1])#设置偏执
	y = tf.matmul(y1, w2) + b2 #计算图
	
	return y#返回值
