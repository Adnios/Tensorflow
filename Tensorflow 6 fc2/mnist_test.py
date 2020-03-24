#验证网络的准确性和泛化性
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
#程序5秒的循环间隔时间
TEST_INTERVAL_SECS = 5

def test(mnist):
    #利用tf.Graph()复现之前定义的计算图
    with tf.Graph().as_default() as g:
        #利用placeholder给训练数据x和标签y_占位
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        #调用mnist_forward文件中的前向传播过程forword()函数
        y = mnist_forward.forward(x, None)
        #实例化具有滑动平均的saver对象，从而在会话被加载时模型中的所有参数被赋值为各自的滑动平均值，增强模型的稳定性
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        #计算模型在测试集上的准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                #加载指定路径下的ckpt
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                #若模型存在，则加载出模型到当前对话，在测试数据集上进行准确率验证，并打印出当前轮数下的准确率
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                #若模型不存在，则打印出模型不存在的提示，从而test()函数完成
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    #加载指定路径下的测试数据集
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()