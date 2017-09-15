import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

## Parameters ##
MAX_STEPS = 3000
BATCH_SIZE = 128
IMG_W = 28
IMG_H = 28
IMG_C = 1
LR = 1e-3
DISPLAY_STEP = 20

## Define DNN class ##
class DNN(object):
    def __init__(self, batch_size, img_w, img_h, img_c, lr):
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c
        self.lr = lr

        ## Placeholders ##
        self.x_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_w * self.img_h * self.img_c])
        self.y_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size])

        ## Network ##
        reshape0 = tf.reshape(self.x_in, shape=[self.batch_size, self.img_w, self.img_h, self.img_c])
        w1 = self.variable_with_weight_loss(shape=[5, 5, 1, 64], stddev=5e-2, wl=0.0)
        kernel1 = tf.nn.conv2d(reshape0, w1, strides=[1, 1, 1, 1], padding='SAME')
        bias1 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))
        conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        w2 = self.variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
        kernel2 = tf.nn.conv2d(norm1, w2, strides=[1, 1, 1, 1], padding='SAME')
        bias2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))
        conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        w3 = self.variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
        bias3 = tf.Variable(tf.zeros(shape=[384], dtype=tf.float32))
        fc3 = tf.nn.relu(tf.matmul(reshape, w3) + bias3)

        w4 = self.variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
        bias4 = tf.Variable(tf.zeros(shape=[192], dtype=tf.float32))
        fc4 = tf.nn.relu(tf.matmul(fc3, w4) + bias4)

        w5 = self.variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
        bias5 = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32))
        logits = tf.matmul(fc4, w5) + bias5

        self.all_loss = self.loss(logits, self.y_in)
        self.top_1 = self.top_1_acc(logits, self.y_in)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.all_loss)

        self.init = tf.global_variables_initializer()

    ## Varable initialization ##
    def variable_with_weight_loss(self, shape, stddev, wl):
        var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        if wl is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var

    ## Overall losses ##
    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'))

    ## Top-1 accuracy ##
    def top_1_acc(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        tmp = tf.nn.in_top_k(logits, labels, 1)
        top_1_acc = tf.reduce_sum(tf.cast(tmp, dtype=tf.float32)) / BATCH_SIZE

        return top_1_acc

## Training ##
dnn = DNN(BATCH_SIZE, IMG_W, IMG_H, IMG_C, LR)
sess = tf.Session()
sess.run(dnn.init)

mnist = input_data.read_data_sets('MNIST', one_hot=False)

for _ in range(MAX_STEPS):

    start_time = time.time()
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run(dnn.train_step, feed_dict={dnn.x_in: batch_x, dnn.y_in: batch_y})

    loss_tr = sess.run(dnn.all_loss, feed_dict={dnn.x_in: batch_x, dnn.y_in: batch_y})
    top_1_tr = sess.run(dnn.top_1, feed_dict={dnn.x_in: batch_x, dnn.y_in: batch_y})
    duration_per_batch = (time.time() - start_time)
    duration_per_sample = duration_per_batch / BATCH_SIZE

    if _ % DISPLAY_STEP == 0:

        print('Batch:', '%04d' % (_ + 1), 'Training loss:', '{:.2f}'.format(loss_tr),
            'Training top-1 acc:', '{:.2f}'.format(top_1_tr), 'Duration / sample:', '{:.2f}'.format(duration_per_sample))

        start_time_te = time.time()
        test_x, test_y = mnist.test.images, mnist.test.labels
        num_iter = int(test_x.shape[0] / BATCH_SIZE)
        loss_te = 0.0
        top_1_te = 0.0
        for i in range(num_iter):
            x = test_x[i*BATCH_SIZE : (i*BATCH_SIZE + BATCH_SIZE)]
            y = test_y[i*BATCH_SIZE : (i*BATCH_SIZE + BATCH_SIZE)]
            loss_te += sess.run(dnn.all_loss, feed_dict={dnn.x_in: x, dnn.y_in: y})
            top_1_te += sess.run(dnn.top_1, feed_dict={dnn.x_in: x, dnn.y_in: y})

        duration_te = (time.time() - start_time)
        duration_per_sample_te = duration_te / test_x.shape[0]
        print('Testing loss:', '{:.2f}'.format(loss_te / num_iter),
              'Testing top-1 acc:', '{:.2f}'.format(top_1_te / num_iter), 'Duration / sample:',
              '{:.2f}'.format(duration_per_sample_te))
