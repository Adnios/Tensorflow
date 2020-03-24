#coding:utf-8
#tensorflow学习笔记(北京大学) tf4_5.py 完全解析神经网络搭建学习
#QQ群：476842922（欢迎加群讨论学习）
#设损失函数 loss=(w+1)^2, 令w初值是常数10。反向传播就是求最优w，即求最小loss对应的w值
#使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更有收敛度。
import tensorflow as tf

LEARNING_RATE_BASE = 0.1 #最初学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减率
LEARNING_RATE_STEP = 1  #喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE

#运行了几轮BATCH_SIZE的计数器，初值给0, 设为不被训练
global_step = tf.Variable(0, trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
#定义待优化参数，初值给10
w = tf.Variable(tf.constant(5, dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)#tf.square()是对a里的每一个元素求平方
#定义反向传播方法    使用minimize()操作，该操作不仅可以优化更新训练的模型参数，也可以为全局步骤(global_step)计数   
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#生成会话，训练40轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()#初始化
    sess.run(init_op)
    for i in range(40):#40次
        sess.run(train_step)#训练
        learning_rate_val = sess.run(learning_rate)#学习率
        global_step_val = sess.run(global_step)#计算获取计数器的值
        w_val = sess.run(w)#计算权重
        loss_val = sess.run(loss)#计算损失函数
        #打印相应数据
        print ("After %s steps: global_step is %f, w is %f, learning rate is %f, loss is %f" % (i, global_step_val, w_val, learning_rate_val, loss_val))
