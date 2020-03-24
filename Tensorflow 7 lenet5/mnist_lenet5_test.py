import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np

TEST_INTERVAL_SECS = 5

#创建一个默认图，在该图中执行以下操作
def test(mnist):
    with tf.Graph().as_default() as g: 
        x = tf.placeholder(tf.float32,[
            mnist.test.num_examples,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS]) 
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
        #训练好的网络，故不使用 dropout
        y = mnist_lenet5_forward.forward(x,False,None)

        ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        #判断预测值和实际值是否相同
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
        ## 求平均得到准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)	
                    # 根据读入的模型名字切分出该模型是属于迭代了多少次保存的
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 
                    reshaped_x = np.reshape(mnist.test.images,(
                    mnist.test.num_examples,
                    mnist_lenet5_forward.IMAGE_SIZE,
                    mnist_lenet5_forward.IMAGE_SIZE,
                    mnist_lenet5_forward.NUM_CHANNELS))
                    #利用多线程提高图片和标签的批获取效率
                    coord = tf.train.Coordinator()#3
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)#4
                    accuracy_score = sess.run(accuracy, feed_dict={x:reshaped_x,y_:mnist.test.labels}) 
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    #关闭线程协调器
                    coord.request_stop()#6
                    coord.join(threads)#7
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS) 

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()