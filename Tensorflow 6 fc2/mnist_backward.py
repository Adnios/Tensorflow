# 2反向传播过程
# 引入tensorflow、input_data、前向传播mnist_forward和os模块
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

# 每轮喂入神经网络的图片数
BATCH_SIZE = 200
# 初始学习率
LEARNING_RATE_BASE = 0.1
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化系数
REGULARIZER = 0.0001
# 训练轮数
STEPS = 50000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径
MODEL_SAVE_PATH = "./model/"
# 模型保存名称
MODEL_NAME = "mnist_model"


def backward(mnist):
    # 用placeholder给训练数据x和标签y_占位
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    # 调用mnist_forward文件中的前向传播过程forword()函数，并设置正则化，计算训练数据集上的预测结果y
    y = mnist_forward.forward(x, REGULARIZER)


    # 当前计算轮数计数器赋值，设定为不可训练类型
    global_step = tf.Variable(0, trainable=False)

    # 调用包含所有参数正则化损失的损失函数loss
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 设定指数衰减学习率learning_rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 使用梯度衰减算法对模型优化，降低损失函数
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 定义参数的滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    # 实例化可还原滑动平均的saver
    # 在模型训练时引入滑动平均可以使模型在测试数据上表现的更加健壮
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 所有参数初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训，加入ckpt操作
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 每次喂入batch_size组（即200组）训练数据和对应标签，循环迭代steps轮
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 将当前会话加载到指定路径
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    # 读入mnist
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    # 反向传播
    backward(mnist)


if __name__ == '__main__':
    main()
