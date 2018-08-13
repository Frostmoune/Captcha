import numpy as np 
import tensorflow as tf
import random
from PIL import Image, ImageDraw
import os

NOW_PATH = str(os.getcwd()).replace('\\', '/') + "/"
CAPTCHA_PATH = NOW_PATH + 'captcha_need/'
LABEL_PATH = NOW_PATH + 'captcha/result.txt'
DIVIDE_PATH = NOW_PATH + 'divide/'
TEST_PATH = NOW_PATH + 'captcha_test/'
MODEL_PATH = NOW_PATH + 'model/'

IMAGE_HEIGHT = 24
IMAGE_WEIGHT = 64
IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WEIGHT

DIVIDE_IMAGE_HEIGHT = 24
DIVIDE_IMAGE_WEIGHT = 16
DIVIDE_IMAGE_SIZE = DIVIDE_IMAGE_HEIGHT * DIVIDE_IMAGE_WEIGHT

CAPTCHA_LENGTH = 4
CAPTCHA_CLASS = 36
LABEL_SIZE = CAPTCHA_CLASS * CAPTCHA_LENGTH

DIVIDE_LABEL_SIZE = 36

BATCH_SIZE = 300

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True
# CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.4
SESS = tf.Session(config = CONFIG)

class CaptchaTensorFlow(object):
    def __init__(self, learn_rate = 0.00025):
        self.learn_rate = learn_rate
    
    # 得到图片路径
    def getImagePath(self, i, base_path = CAPTCHA_PATH):
        image_path = base_path + "0"
        if i < 100:
            image_path += "0"
        if i < 10:
            image_path += "0"
        image_path += str(i) + '.png'
        return image_path
    
    # 得到图片的像素
    def getImagePixel(self, image):
        try:
            res = np.array(image).flatten()
        except Exception as e:
            print(e)
        return res
    
    # 计算labels
    def calLabels(self):
        self.labels = np.zeros((self.sample_num, LABEL_SIZE))
        fp = open(LABEL_PATH)
        row = 0
        for line in fp.readlines():
            now_line = str(line).strip()
            col_labels = np.zeros((1, LABEL_SIZE))
            for i in range(CAPTCHA_LENGTH):
                if str.isdigit(str(now_line[i])):
                    col_labels[0, i * CAPTCHA_CLASS + int(str(now_line[i]))] = 1
                else:
                    col_labels[0, i * CAPTCHA_CLASS + ord(str(now_line[i])) - 87] = 1
            self.labels[row, :] = col_labels
            row += 1

    # 计算features
    def calFeatures(self):
        self.sample_num = len(os.listdir(CAPTCHA_PATH))
        self.features = np.zeros((self.sample_num, IMAGE_SIZE))
        for image_num in range(self.sample_num):
            now_image = Image.open(self.getImagePath(image_num + 1))
            self.features[image_num, :] = self.getImagePixel(now_image)
    
    # 得到验证码种类
    def getCaptchaClass(self):
        self.captcha_class = []
        for i in range(36):
            if i < 10:
                self.captcha_class.append(str(i))
            else:
                self.captcha_class.append(chr(i + 87))
    
    # 得到分割验证码的features和labels
    def getDivideFeaturesAndLabels(self):
        self.getCaptchaClass()
        self.features = np.zeros((4000, DIVIDE_IMAGE_SIZE))
        self.labels = np.zeros((4000, DIVIDE_LABEL_SIZE))
        self.sample_num = 0
        for now_class in self.captcha_class:
            now_path = DIVIDE_PATH + now_class
            for x in os.listdir(now_path):
                if x[-4:] != '.png':
                    continue
                png_path = now_path + '/' + x
                now_image = Image.open(png_path)
                self.features[self.sample_num, :] = self.getImagePixel(now_image)
                if str.isdigit(now_class):
                    self.labels[self.sample_num, int(now_class)] = 1
                else:
                    self.labels[self.sample_num, ord(now_class) - 87] = 1
                self.sample_num += 1
        self.features = self.features[:self.sample_num, :]
        self.labels = self.labels[:self.sample_num, :]

    # 得到一个训练的batch
    def getBatch(self, batch_size = BATCH_SIZE): 
        now_list = list(range(self.sample_num))
        random.shuffle(now_list)
        now_list = now_list[:batch_size]
        batch_x = self.features[now_list, :]
        batch_y = self.labels[now_list, :]
        return batch_x, batch_y
    
    # 初始化权重
    def weightVariable(self, shape, name = ""):
        initial = tf.truncated_normal(shape, mean = 0.0, stddev = 0.1)
        if name == "":
            return tf.Variable(initial)
        return tf.Variable(initial, name = name)

    # 初始化偏置
    def biasVariable(self, shape, name = ""):
        initial = tf.truncated_normal(shape = shape, stddev = 1.0)
        if name == "":
            return tf.Variable(initial)
        return tf.Variable(initial, name = name)
    
    # debug
    def debug(self, is_divide = 0):
        if not is_divide:
            now_height = IMAGE_HEIGHT
            now_weight = IMAGE_WEIGHT
        else:
            now_height = DIVIDE_IMAGE_HEIGHT
            now_weight = DIVIDE_IMAGE_WEIGHT

        num = 0
        while True:
            for i in range(now_height):
                for j in range(now_weight):
                    print(int(self.features[num, i * now_weight + j]), end = '')
                print()
            print("\nLabel:", end = "")
            if not is_divide:
                print(self.bitsToResult(self.labels[num, :].reshape(1, LABEL_SIZE)))
            else:
                print(self.bitsToResultDivide(self.labels[num, :].reshape(1, DIVIDE_LABEL_SIZE)))
            input()
            num += 1
            
    # 卷积
    def conv2d(self, x, W, name = ""):
        if name == "":
            return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME', name = name)

    # 池化
    def maxPool2x2(self, x, name = ""):
        if name == "":
            return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)
    
    # 初始化每一层
    def initCNNConv(self, pre_conv, channels_in, channels_out, index):
        with tf.name_scope('conv' + str(index)):
            # 计算权重（3,3表示卷积核为3*3的，channels_in表示输入通道数，channels_out输出通道数）
            W_conv = self.weightVariable([3, 3, channels_in, channels_out], name = 'W_conv' + str(index))
            W_max = tf.argmax(W_conv, )
            # 计算偏置
            b_conv = self.biasVariable([channels_out], name = 'b_conv' + str(index))
            # 卷积层
            h_conv = tf.nn.relu(tf.nn.bias_add(self.conv2d(pre_conv, W_conv), b_conv), name = 'h_conv' + str(index))
            # 池化层
            h_pool = self.maxPool2x2(h_conv, name = 'h_pool' + str(index))
            # 加入一层dropout
            # h_drop = tf.nn.dropout(h_pool, self.keep_prob, name = 'h_drop' + str(index))
            tf.summary.histogram('W_conv', W_conv)
            tf.summary.histogram('b_conv', b_conv)
            tf.summary.histogram('h_conv', h_conv)
            tf.summary.histogram('h_pool', h_pool)
            # tf.summary.histogram('h_drop', h_drop)
            return h_pool
    
    # CNN初始化
    def initCNN(self, is_divide = 0):
        if not is_divide:
            feature_size = IMAGE_SIZE
            feature_weight = IMAGE_WEIGHT
            feature_height = IMAGE_HEIGHT
            label_size = LABEL_SIZE
        else:
            feature_size = DIVIDE_IMAGE_SIZE
            feature_weight = DIVIDE_IMAGE_WEIGHT
            feature_height = DIVIDE_IMAGE_HEIGHT
            label_size = DIVIDE_LABEL_SIZE

        self.X = tf.placeholder(tf.float32, [None, feature_size], name = 'features')
        self.Y = tf.placeholder(tf.float32, [None, label_size], name = 'labels')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        # 转化成4d
        x_image = tf.reshape(self.X, [-1, feature_height, feature_weight, 1], name = 'x_image')
        tf.summary.image('x_image', x_image, 4)

        # 第一层卷积
        h_pool1 = self.initCNNConv(x_image, 1, 64, 1)

        # 第二层卷积
        h_pool2 = self.initCNNConv(h_pool1, 64, 128, 2)

        # 第三层卷积
        h_pool3 = self.initCNNConv(h_pool2, 128, 256, 3)

        # 全连接层
        with tf.name_scope('flat'):
            W_fc1 = self.weightVariable([feature_size // 64 * 256, 1080], name = 'W_fc1') # feature_size / 64 * 256
            b_fc1 = self.biasVariable([1080], name = 'b_fc1')
            h_pool3_flat = tf.reshape(h_pool3, [-1, feature_size // 64 * 256], name = 'h_pool3_flat')
            h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, W_fc1), b_fc1), name = 'h_fc1')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob, name = 'h_fc1_drop')
            tf.summary.histogram('W_fc1', W_fc1)
            tf.summary.histogram('b_fc1', b_fc1)
            tf.summary.histogram('h_pool3_flat', h_pool3_flat)
            tf.summary.histogram('h_fc1', h_fc1)
            tf.summary.histogram('h_fc1_drop', h_fc1_drop)

        # 输出层
        with tf.name_scope('output'):
            W_fc2 = self.weightVariable([1080, label_size], name = 'W_fc2')
            b_fc2 = self.biasVariable([label_size], name = 'b_fc2')
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name = 'y_conv')
            tf.summary.histogram('W_fc2', W_fc2)
            tf.summary.histogram('b_fc2', b_fc2)
            tf.summary.histogram('y_conv', y_conv)
        
        tf.add_to_collection('pred_network', y_conv)

        return y_conv
    
    # 训练
    def train(self, is_divide = 0):
        if not is_divide:
            self.calFeatures()
            self.calLabels()
        else:
            self.getDivideFeaturesAndLabels() 
        # self.debug(is_divide)
        y_conv = self.initCNN(is_divide)

        # 损失函数
        with tf.name_scope('loss'):
            loss = -tf.reduce_sum(self.Y * tf.log(y_conv + 1e-10)) # 加1e-10是必要的
        tf.summary.scalar('loss', loss)

        # optimizer 为了加快训练
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learn_rate).minimize(loss)

        if not is_divide:
            predict = tf.reshape(y_conv, [-1, CAPTCHA_LENGTH, CAPTCHA_CLASS])
            max_idx_p = tf.argmax(predict, 2)
            max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, CAPTCHA_LENGTH, CAPTCHA_CLASS]), 2)
            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(max_idx_p, max_idx_l)
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', accuracy)
        else:
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

        saver = tf.train.Saver()
        SESS.run(tf.global_variables_initializer())

        step = 0
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('captcha_logs')
        summary_writer.add_graph(SESS.graph)
        while step < 2001:
            batch_x, batch_y = self.getBatch()
            now_summary, now_loss, _ = SESS.run([merged_summary, loss, optimizer], feed_dict = {self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.9})
            summary_writer.add_summary(now_summary, step)
            print("第%d步完成"%step, end = ",loss：")
            print(now_loss)

            # 每50 step计算一次准确率
            if step % 50 == 0:
                batch_x_test, batch_y_test = self.getBatch(150)
                now_summary, acc = SESS.run([merged_summary, accuracy], feed_dict = {self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                print("第%d步准确率为："%step, end = "")
                print(acc)
                summary_writer.add_summary(now_summary, step)
                
            step += 1
        saver.save(SESS, "./model/capcha_model.ckpt", global_step = step)
        
    # 结果转换为文本
    def bitsToResult(self, y):
        res = ""
        for i in range(CAPTCHA_LENGTH):
            for j in range(CAPTCHA_CLASS):
                if y[0, i * CAPTCHA_CLASS + j] >= 1:
                    if j < 10:
                        res += str(j)
                    else:
                        res += chr(j + 87)
        return res
    
    # 结果转换为文本
    def bitsToResultDivide(self, y):
        res = ""
        for i in range(DIVIDE_LABEL_SIZE):
            if y[0, i] >= 1:
                if i < 10:
                    res += str(i)
                else:
                    res += chr(i + 87)
        return res
    
    # 得到测试结果
    def getPredictResult(self, y):
        res = ""
        for i in range(CAPTCHA_LENGTH):
            now_list = y[0, i * CAPTCHA_CLASS : (i + 1) * CAPTCHA_CLASS].tolist()
            j = now_list.index(max(now_list))
            if j < 10:
                res += str(j)
            else:
                res += chr(j + 87)
        return res
    
    # 得到测试结果（divide）
    def getDividePredictResult(self, y):
        now_list = y[0].tolist()
        j = now_list.index(max(now_list))
        if j < 10:
            res = str(j)
        else:
            res = chr(j + 87)
        return res
    
    # 得到测试样本
    def getTestFeature(self, is_divide = 0):
        if not is_divide:
            self.test_num = (len(os.listdir(TEST_PATH)) - 1) // 2
            self.test_feature = np.zeros((self.test_num, IMAGE_SIZE))
            num = self.test_num
        else:
            self.test_num = (len(os.listdir(TEST_PATH)) - 1) // 2 * CAPTCHA_LENGTH
            self.test_feature = np.zeros((self.test_num, DIVIDE_IMAGE_SIZE))
            num = self.test_num // CAPTCHA_LENGTH

        for i in range(num):
            now_path = self.getImagePath(i + 1, TEST_PATH)[:-4] + '_test.png'
            now_image = Image.open(now_path)
            if not is_divide:
                self.test_feature[i, :] = self.getImagePixel(now_image)
            else:
                for j in range(CAPTCHA_LENGTH):
                    divide_image = now_image.crop((j * DIVIDE_IMAGE_WEIGHT, 0, (j + 1) * DIVIDE_IMAGE_WEIGHT, DIVIDE_IMAGE_HEIGHT))
                    self.test_feature[i * CAPTCHA_LENGTH + j, :] = self.getImagePixel(divide_image)
        
    # 得到测试label
    def getTestLabel(self, is_divide = 0):
        if not is_divide:
            now_label_size = LABEL_SIZE
        else:
            now_label_size = DIVIDE_LABEL_SIZE
        self.test_labels = np.zeros((self.test_num, now_label_size))

        fp = open(TEST_PATH + 'result.txt')
        row = 0
        for line in fp.readlines():
            now_line = str(line).strip()
            if not is_divide:
                col_labels = np.zeros((1, now_label_size))
                for i in range(CAPTCHA_LENGTH):
                    if str.isdigit(str(now_line[i])):
                        col_labels[0, i * CAPTCHA_CLASS + int(str(now_line[i]))] = 1
                    else:
                        col_labels[0, i * CAPTCHA_CLASS + ord(str(now_line[i])) - 87] = 1
                self.test_labels[row, :] = col_labels
                row += 1
            else:
                for i in range(CAPTCHA_LENGTH):
                    col_labels = np.zeros((1, now_label_size))
                    if str.isdigit(str(now_line[i])):
                        col_labels[0, int(str(now_line[i]))] = 1
                    else:
                        col_labels[0, ord(str(now_line[i])) - 87] = 1
                    self.test_labels[row, :] = col_labels
                    row += 1
    
    # 加载训练模型
    def loadModel(self):
        model_list = os.listdir(MODEL_PATH)
        meta_path = MODEL_PATH
        now_model_path = MODEL_PATH
        for x in model_list:
            if str(x)[-5:] == '.meta':
                meta_path += str(x)
                now_model_path += str(x)[:-5]
                break

        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(SESS, now_model_path)

        self.test_y = tf.get_collection('pred_network')[0]

        graph = tf.get_default_graph()

        self.test_x = graph.get_operation_by_name('features').outputs[0]
        self.test_keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    # 测试
    def Test(self, is_divide = 0):
        self.getTestFeature(is_divide)
        self.getTestLabel(is_divide)
        self.loadModel()

        if not is_divide:
            feature_size = IMAGE_SIZE
            label_size = LABEL_SIZE
            loop_num = self.test_num
        else:
            feature_size = DIVIDE_IMAGE_SIZE
            label_size = DIVIDE_LABEL_SIZE
            loop_num = self.test_num // CAPTCHA_LENGTH

        true_num = 0
        total_num = 0
        for i in range(loop_num):
            if not is_divide:
                res = SESS.run(self.test_y, feed_dict = {self.test_x: self.test_feature[i, :].reshape(1, feature_size), self.test_keep_prob: 1.})
                pred_res = self.getPredictResult(res[0].reshape(1, label_size))
                true_res = self.bitsToResult(self.test_labels[i, :].reshape(1, label_size))
            else:
                pred_res, true_res = "", ""
                for j in range(CAPTCHA_LENGTH):
                    res = SESS.run(self.test_y, feed_dict = {self.test_x: self.test_feature[i * CAPTCHA_LENGTH + j, :].reshape(1, feature_size), self.test_keep_prob: 1.})
                    pred_res += self.getDividePredictResult(res[0].reshape(1, label_size))
                    true_res += self.bitsToResultDivide(self.test_labels[i * CAPTCHA_LENGTH + j, :].reshape(1, label_size))

            print("正确结果：%s\t预测结果：%s\t"%(true_res, pred_res), end = "")
            if pred_res == true_res:
                true_num += 1
                print("  正确")
            else:
                print("  错误")
            total_num += 1
        print("正确率：", true_num / total_num)


if __name__ == '__main__':
    train_nn = CaptchaTensorFlow()
    choice = input("1、Train\n2、Test\n")
    if choice == '1':
        train_nn.train(is_divide = 1)
    else:
        train_nn.Test(is_divide = 1)