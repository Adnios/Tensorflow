#coding:utf-8
#tensorflow学习笔记(北京大学) tf4_2.py 完全解析神经网络搭建学习
#QQ群：476842922（欢迎加群讨论学习）
#酸奶成本1元， 酸奶利润9元
#预测少了损失大，故不要预测少，故生成的模型会多预测一些
#0导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455#随机种子
COST = 1#花费
PROFIT = 9#成本

rdm = np.random.RandomState(SEED)#基于seed产生随机数
X = rdm.rand(32,2)#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

#1定义神经网络的输入、参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))#占位
y_ = tf.placeholder(tf.float32, shape=(None, 1))#占位
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))#正态分布
y = tf.matmul(x, w1)#点积

#2定义损失函数及反向传播方法。
# 定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。
#tf.where:如果condition对应位置值为True那么返回Tensor对应位置为x的值，否则为y的值.
#where(condition, x=None, y=None,name=None)
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ - y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)#随机梯度下降

#3生成会话，训练STEPS轮。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()#初始化
    sess.run(init_op)#初始化
    STEPS = 3000
    for i in range(STEPS):#三千轮
        start = (i*BATCH_SIZE) % 32  #8个数据  为一个数据块输出
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE  #[i:i+8]
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})#训练
        if i % 500 == 0:#每500轮打印输出
            print("After %d training steps, w1 is: " % (i))#打印i
            print(sess.run(w1), "\n")#打印w1
    print("Final w1 is: \n", sess.run(w1))#最终打印w1
