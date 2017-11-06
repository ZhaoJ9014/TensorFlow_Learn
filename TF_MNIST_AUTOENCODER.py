import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

## Training data
mnist = input_data.read_data_sets('./tmp/mnist/', one_hot = True)

## Hyperparameters
LR = 0.01
EPOCHES = 20
BATCH_SIZE = 256
DISPLAY_STEP = 1
EXAMPLES_TO_SHOW = 10

N_INPUT = 784
N_HIDDEN_1 = 256
N_HIDDEN_2 = 128

## Placeholders
X = tf.placeholder(dtype = tf.float32, shape = [None, N_INPUT])

## Weights & Biases
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_1])),
    'encoder_h2': tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2])),
    'decoder_h1': tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_1])),
    'decoder_h2': tf.Variable(tf.random_normal([N_HIDDEN_1, N_INPUT]))
}

biases = {
    'encoder_h1': tf.Variable(tf.random_normal([N_HIDDEN_1])),
    'encoder_h2': tf.Variable(tf.random_normal([N_HIDDEN_2])),
    'decoder_h1': tf.Variable(tf.random_normal([N_HIDDEN_1])),
    'decoder_h2': tf.Variable(tf.random_normal([N_INPUT]))
}

## Define network
def encoder(x):
    x = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, weights['encoder_h1']), biases['encoder_h1']))
    x = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, weights['encoder_h2']), biases['encoder_h2']))
    return x

def decoder(x):
    x = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, weights['decoder_h1']), biases['decoder_h1']))
    x = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, weights['decoder_h2']), biases['decoder_h2']))
    return x

## Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_gt = X

loss = tf.reduce_mean(tf.pow(y_gt - y_pred, 2))
train_op = tf.train.RMSPropOptimizer(LR).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    BATCH_NUM = int(mnist.train.num_examples / BATCH_SIZE)
    # Training
    for e in range(EPOCHES):
        for b in range(BATCH_NUM):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict = {X: batch_x})
            tr_loss = sess.run(loss, feed_dict={X: batch_x})
        if e % DISPLAY_STEP == 0:
            print("Epoch:", '%04d' % (e + 1), "loss:", "{:.9f}".format(tr_loss))
    print("Optimization finished!")
    # Testing
    pred = sess.run(y_pred, feed_dict = {X: mnist.test.images[:EXAMPLES_TO_SHOW]})
    f, a = plt.subplots(2, 10, figsize = (10, 2))
    for i in range(EXAMPLES_TO_SHOW):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(pred[i], (28, 28)))
    f.show()
    plt.show()
    plt.waitforbuttonpress()
