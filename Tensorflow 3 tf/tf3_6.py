#coding:utf-8
#0导入模块，生成模拟数据集。
#tensorflow学习笔记(北京大学) tf3_6.py 完全解析神经网络搭建学习
#QQ群：476842922（欢迎加群讨论学习
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X:\n",X)
print("Y_:\n",Y_)
x = tf.placeholder(tf.float32, shape=(None, 2))#用placeholder实现输入定义
y_= tf.placeholder(tf.float32, shape=(None, 1))#用placeholder实现占位
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))#正态分布随机数
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))#正态分布随机数
a = tf.matmul(x, w1)#点积
y = tf.matmul(a, w2)#点积

#2定义损失函数及反向传播方法。
loss_mse = tf.reduce_mean(tf.square(y-y_)) 
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()#初始化
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值。
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")
    
    # 训练模型。
    STEPS = 3000
    for i in range(STEPS):#3000轮
        start = (i*BATCH_SIZE) % 32 #i*8%32
        end = start + BATCH_SIZE    #i*8%32+8
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))
    
    # 输出训练后的参数取值。
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
#，只搭建承载计算过程的
#计算图，并没有运算，如果我们想得到运算结果就要用到“会话 Session()”了。 
#√会话（Session）： 执行计算图中的节点运算  
    print("w1:\n", w1)
    print("w2:\n", w2)
"""
X:
[[ 0.83494319  0.11482951]
 [ 0.66899751  0.46594987]
 [ 0.60181666  0.58838408]
 [ 0.31836656  0.20502072]
 [ 0.87043944  0.02679395]
 [ 0.41539811  0.43938369]
 [ 0.68635684  0.24833404]
 [ 0.97315228  0.68541849]
 [ 0.03081617  0.89479913]
 [ 0.24665715  0.28584862]
 [ 0.31375667  0.47718349]
 [ 0.56689254  0.77079148]
 [ 0.7321604   0.35828963]
 [ 0.15724842  0.94294584]
 [ 0.34933722  0.84634483]
 [ 0.50304053  0.81299619]
 [ 0.23869886  0.9895604 ]
 [ 0.4636501   0.32531094]
 [ 0.36510487  0.97365522]
 [ 0.73350238  0.83833013]
 [ 0.61810158  0.12580353]
 [ 0.59274817  0.18779828]
 [ 0.87150299  0.34679501]
 [ 0.25883219  0.50002932]
 [ 0.75690948  0.83429824]
 [ 0.29316649  0.05646578]
 [ 0.10409134  0.88235166]
 [ 0.06727785  0.57784761]
 [ 0.38492705  0.48384792]
 [ 0.69234428  0.19687348]
 [ 0.42783492  0.73416985]
 [ 0.09696069  0.04883936]]
Y_:
[[1], [0], [0], [1], [1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [0], [0], [1], [1], [0], [1], [0], [1], [1], [1], [1], [1], [0], [1]]
w1:
[[-0.81131822  1.48459876  0.06532937]
 [-2.4427042   0.0992484   0.59122431]]
w2:
[[-0.81131822]
 [ 1.48459876]
 [ 0.06532937]]


After 0 training step(s), loss_mse on all data is 5.13118
After 500 training step(s), loss_mse on all data is 0.429111
After 1000 training step(s), loss_mse on all data is 0.409789
After 1500 training step(s), loss_mse on all data is 0.399923
After 2000 training step(s), loss_mse on all data is 0.394146
After 2500 training step(s), loss_mse on all data is 0.390597


w1:
[[-0.70006633  0.9136318   0.08953571]
 [-2.3402493  -0.14641267  0.58823055]]
w2:
[[-0.06024267]
 [ 0.91956186]
 [-0.0682071 ]]
"""

