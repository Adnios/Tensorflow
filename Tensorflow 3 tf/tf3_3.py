#coding:utf-8
#两层简单神经网络（全连接）
#tensorflow学习笔记(北京大学) tf3_3.py 完全解析
#QQ群：476842922（欢迎加群讨论学习）
import tensorflow as tf

#表示生成正态分布随机数，形状两行三列，标准差是 2，均值是 0，随机种子是 1。 
x = tf.constant([[0.7, 0.5]])#定义一个张量等于[1.0,2.0]
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))# 生成正态分布随机数[2, 3]
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))# 生成正态分布随机数[3, 1]

#定义前向传播过程
a = tf.matmul(x, w1)#点积
y = tf.matmul(a, w2)#点积

#用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()#变量初始化
    sess.run(init_op)#计算
    print"y in tf3_3.py is:\n",sess.run(y) #打印计算结果

'''
y in tf3_3.py is : 
[[3.0904665]]
'''
# √神经网络的实现过程：
# 1、准备数据集，提取特征，作为输入喂给神经网络（Neural Network，NN）
# 2、搭建 NN 结构，从输入到输出（先搭建计算图，再用会话执行）
# （ NN 前向传播算法        计算输出）
# 3、大量特征数据喂给 NN，迭代优化 NN 参数
#        （ NN 反向传播算法         优化参数训练模型）
# 4、使用训练好的模型预测和分类
