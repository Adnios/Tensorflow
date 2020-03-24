import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np

# batch的数量
BATCH_SIZE = 100
# 初始学习率
LEARNING_RATE_BASE = 0.005
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化
REGULARIZER = 0.0001
# 最大迭代次数
STEPS = 50000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径
MODEL_SAVE_PATH = "./model/"
# 模型名称
MODEL_NAME = "mnist_model"


def backward(mnist):
    # 卷积层输入为四阶张量
    # 第一阶表示每轮喂入的图片数量，第二阶和第三阶分别表示图片的行分辨率和列分辨率，第四阶表示通道数
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        mnist_lenet5_forward.IMAGE_SIZE,
        mnist_lenet5_forward.IMAGE_SIZE,
        mnist_lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    # 前向传播过程
    y = mnist_lenet5_forward.forward(x, True, REGULARIZER)
    # 声明一个全局计数器
    global_step = tf.Variable(0, trainable=False)
    # 对网络最后一层的输出y做softmax，求取输出属于某一类的概率
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 向量求均值
    cem = tf.reduce_mean(ce)
    # 正则化的损失值
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    # 梯度下降算法的优化器
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
    # 采用滑动平均的方法更新参数
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)


    ema_op = ema.apply(tf.trainable_variables())
    # 将train_step和ema_op两个训练操作绑定到train_op上
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 实例化一个保存和恢复变量的saver
    saver = tf.train.Saver()
    # 创建一个会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 通过 checkpoint 文件定位到最新保存的模型，若文件存在，则加载最新的模型
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            # 读取一个batch数据，将输入数据xs转成与网络输入相同形状的矩阵
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.NUM_CHANNELS))
            # 读取一个batch数据，将输入数据xs转成与网络输入相同形状的矩阵
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()