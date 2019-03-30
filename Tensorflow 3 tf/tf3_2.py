#coding=utf-8
import tensorflow as tf
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
y = tf.matmul(x,w)#计算图（Graph）：
print y
with tf.Session() as sess:
    print sess.run(y)# 执行计算图中的节点运算。 


