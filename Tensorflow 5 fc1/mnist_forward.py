# 1前向传播过程
import tensorflow as tf

# 网络输入节点为784个（代表每张输入图片的像素个数）
INPUT_NODE = 784
# 输出节点为10个（表示输出为数字0-9的十分类）
OUTPUT_NODE = 10
# 隐藏层节点500个
LAYER1_NODE = 500


def get_weight(shape, regularizer):
    # 参数满足截断正态分布，并使用正则化，
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # w = tf.Variable(tf.random_normal(shape,stddev=0.1))
    # 将每个参数的正则化损失加到总损失中
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    # 初始化的一维数组，初始化值为全 0
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    # 由输入层到隐藏层的参数w1形状为[784,500]
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    # 由输入层到隐藏的偏置b1形状为长度500的一维数组，
    b1 = get_bias([LAYER1_NODE])
    # 前向传播结构第一层为输入 x与参数 w1矩阵相乘加上偏置 b1 ，再经过relu函数 ，得到隐藏层输出 y1。
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # 由隐藏层到输出层的参数w2形状为[500,10]
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    # 由隐藏层到输出的偏置b2形状为长度10的一维数组
    b2 = get_bias([OUTPUT_NODE])
    # 前向传播结构第二层为隐藏输出 y1与参 数 w2 矩阵相乘加上偏置 矩阵相乘加上偏置 b2，得到输出 y。
    # 由于输出 。由于输出 y要经过softmax oftmax 函数，使其符合概率分布，故输出y不经过 relu函数
    y = tf.matmul(y1, w2) + b2
    return y