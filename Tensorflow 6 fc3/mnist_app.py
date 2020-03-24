import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(testPicArr):
    # 利用tf.Graph()复现之前定义的计算图
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        # 调用mnist_forward文件中的前向传播过程forword()函数
        y = mnist_forward.forward(x, None)
        # 得到概率最大的预测值
        preValue = tf.argmax(y, 1)

        # 实例化具有滑动平均的saver对象
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            # 通过ckpt获取最新保存的模型
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


# 预处理，包括resize，转变灰度图，二值化
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    # 对图片做二值化处理（这样以滤掉噪声，另外调试中可适当调节阈值）
    threshold = 50
    # 模型的要求是黑底白字，但输入的图是白底黑字，所以需要对每个像素点的值改为255减去原值以得到互补的反色。
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    # 把图片形状拉成1行784列，并把值变为浮点型（因为要求像素点是0-1 之间的浮点数）
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    # 接着让现有的RGB图从0-255之间的数变为0-1之间的浮点数
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def application():
    # 输入要识别的几张图片
    testNum = int(input("input the number of test pictures:"))
    for i in range(testNum):
        # 给出待识别图片的路径和名称
        testPic = input("the path of test picture:")
        # 图片预处理
        testPicArr = pre_pic(testPic)
        # 获取预测结果
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)


def main():
    application()


if __name__ == '__main__':
    main()