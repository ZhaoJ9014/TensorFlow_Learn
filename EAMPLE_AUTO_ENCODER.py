import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## Import MNIST data ##
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST', one_hot=True)

## Parameters ##
LR = 0.01
NB_EPOCH = 500
BATCH_SIZE = 256
EXAMPLE_TO_SHOW = 10
INPUT_SIZE = 784
OUTPUT_SIZE = 784

## Auto-encoder ##
class AUTO_ENCODER(object):
    def __init__(self, lr, batch_size, input_size, output_size):
        self.lr = lr
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size

        self.xs = tf.placeholder(tf.float32, [None, input_size], name='xs')
        with tf.variable_scope('encoder'):
            self.add_encoder()
        with tf.variable_scope('decoder'):
            self.add_decoder()

        self.compute_loss()
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        ## add_encoder ##
    def add_encoder(self):
        W_e1 = self._weight_variable(shape=[self.input_size, 128], name='weights_e1')
        b_e1 = self._bias_variable(shape=[128], name='bias_e1')
        e1_out = tf.nn.sigmoid(tf.matmul(self.xs, W_e1) + b_e1)
        W_e2 = self._weight_variable(shape=[128, 64], name='weights_e2')
        b_e2 = self._bias_variable(shape=[64], name='bias_e2')
        e2_out = tf.nn.sigmoid(tf.matmul(e1_out, W_e2) + b_e2)
        W_e3 = self._weight_variable(shape=[64, 10], name='weights_e3')
        b_e3 = self._bias_variable(shape=[10], name='bias_e3')
        e3_out = tf.nn.sigmoid(tf.matmul(e2_out, W_e3) + b_e3)
        W_e4 = self._weight_variable(shape=[10, 2], name='weights_e4')
        b_e4 = self._bias_variable(shape=[2], name='bias_e4')
        self.e4_out = tf.matmul(e3_out, W_e4) + b_e4


    def add_decoder(self):
        W_d1 = self._weight_variable(shape=[2, 10], name='weights_d1')
        b_d1 = self._bias_variable(shape=[10], name='bias_d1')
        d1_out = tf.nn.sigmoid(tf.matmul(self.e4_out, W_d1) + b_d1)
        W_d2 = self._weight_variable(shape=[10, 64], name='weights_d2')
        b_d2 = self._bias_variable(shape=[64], name='bias_d2')
        d2_out = tf.nn.sigmoid(tf.matmul(d1_out, W_d2) + b_d2)
        W_d3 = self._weight_variable(shape=[64, 128], name='weights_d3')
        b_d3 = self._bias_variable(shape=[128], name='bias_d3')
        d3_out = tf.nn.sigmoid(tf.matmul(d2_out, W_d3) + b_d3)
        W_d4 = self._weight_variable(shape=[128, 784], name='weights_d4')
        b_d4 = self._bias_variable(shape=[784], name='bias_d4')
        self.pred = tf.nn.sigmoid(tf.matmul(d3_out, W_d4) + b_d4)

    def compute_loss(self):
        self.loss = tf.reduce_mean(tf.square((self.pred - self.xs)))
        tf.summary.scalar('loss', self.loss)

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

if __name__ == '__main__':
    model = AUTO_ENCODER(LR, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)

    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(NB_EPOCH):
        X_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss, pred = sess.run(
            [model.train_step, model.loss, model.pred],
            feed_dict={model.xs: X_batch})

        if i % 20 == 0:
            print('loss: ', loss)
            result = sess.run(merged, feed_dict={model.xs: X_batch})
            writer.add_summary(result, i)

    # encode_decode = sess.run(
    #         model.pred, feed_dict={model.xs: mnist.test.images[:EXAMPLE_TO_SHOW]})
    # f, a = plt.subplots(2, EXAMPLE_TO_SHOW, figsize=(10, 2))
    # for i in range(EXAMPLE_TO_SHOW):
    #     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # plt.show()

    encode_decode = sess.run(model.e4_out, feed_dict={model.xs: mnist.test.images})
    plt.scatter(encode_decode[:, 0], encode_decode[:, 1], c = mnist.test.labels)
    plt.show()
