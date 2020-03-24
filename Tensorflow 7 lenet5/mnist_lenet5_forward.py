import tensorflow as tf

# 每张图片分辨率为28*28
IMAGE_SIZE = 28
# Mnist数据集为灰度图，故输入图片通道数NUM_CHANNELS取值为1
NUM_CHANNELS = 1
# 第一层卷积核大小为5
CONV1_SIZE = 5
# 卷积核个数为32
CONV1_KERNEL_NUM = 32
# 第二层卷积核大小为5
CONV2_SIZE = 5
# 卷积核个数为64
CONV2_KERNEL_NUM = 64
# 全连接层第一层为 512 个神经元
FC_SIZE = 512
# 全连接层第二层为 10 个神经元
OUTPUT_NODE = 10


# 权重w计算
def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 偏置b计算
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


# 卷积层计算
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 最大池化层计算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x, train, regularizer):
    # 实现第一层卷积
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    # 非线性激活
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 最大池化
    pool1 = max_pool_2x2(relu1)

    # 实现第二层卷积
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 获取一个张量的维度
    pool_shape = pool2.get_shape().as_list()
    # pool_shape[1] 为长 pool_shape[2] 为宽 pool_shape[3]为高
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 得到矩阵被拉长后的长度，pool_shape[0]为batch值
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 实现第三层全连接层
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层输出使用dropout
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    # 实现第四层全连接层
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y 