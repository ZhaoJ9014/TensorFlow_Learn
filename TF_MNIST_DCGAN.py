## DC-GAN ##
import tensorflow as tf
import numpy as np
from convert2tfrecord import convert2tfrecord
from extract_tfrecord import extract_tfrecord
from PIL import Image

## Convert training and testing data into TFRecord ##
convert2tfrecord('../Data/train/', '../List/train.txt', 28, 28, 1, '../Data/TF_MNIST_train.tfrecords')

## Intialize hyperparameters ##
LR = 2e-4
BATCH_SIZE = 64
Z_DIM = 100
IMG_W = 28
IMG_H = 28
IMG_C = 1
LEAK = 0.2

## Main Network ##

# Define placeholders
z = tf.placeholder(dtype = tf.float32, shape = [None, Z_DIM], name = 'input_noise')
X_real = tf.placeholder(dtype = tf.float32, shape = [None, IMG_W, IMG_H, IMG_C], name = 'input_img')

# Define generator
def generator(z):

    with tf.variable_scope('generator') as scope:
        # Project and reshape input
        l1 = tf.layers.dense(z, 7*7*256, name = 'g_l1_FC')
        l1 = tf.reshape(l1, shape = [-1, 7, 7, 256], name = 'g_l1_RS')
        l1 = tf.layers.batch_normalization(l1, name = 'g_l1_BN')
        l1 = tf.nn.relu(l1, name = 'g_l1_RELU')
        # Deconvs
        l2 = tf.layers.conv2d_transpose(l1, filters = 128, kernel_size = [5, 5], strides = [2, 2], padding = 'same',
                                        name = 'g_l2_DECONV')
        l2 = tf.layers.batch_normalization(l2, name='g_l2_BN')
        l2 = tf.nn.relu(l2, name='g_l2_RELU')

        l3 = tf.layers.conv2d_transpose(l2, filters = 1, kernel_size = [5, 5], strides = [2, 2], padding = 'same',
                                        name = 'g_l3_DECONV')
        # Ouput
        g_pred = tf.nn.tanh(l3, name = 'g_l3_TANH')

        return g_pred

# Define Sampler
def sampler(z):

    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        # Project and reshape input
        l1 = tf.layers.dense(z, 7 * 7 * 256, name='g_l1_FC')
        l1 = tf.reshape(l1, shape=[-1, 7, 7, 256], name='g_l1_RS')
        l1 = tf.layers.batch_normalization(l1, name='g_l1_BN', trainable = False)
        l1 = tf.nn.relu(l1, name='g_l1_RELU')
        # Deconvs
        l2 = tf.layers.conv2d_transpose(l1, filters=128, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                        name='g_l2_DECONV')
        l2 = tf.layers.batch_normalization(l2, name='g_l2_BN', trainable = False)
        l2 = tf.nn.relu(l2, name='g_l2_RELU')

        l3 = tf.layers.conv2d_transpose(l2, filters=1, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                        name='g_l3_DECONV')
        # Ouput
        generated = tf.nn.tanh(l3, name='g_l3_TANH')
        generated = (generated + 1.) * 127.5

        return generated

# Define Discriminator
def discriminator(X, reuse = False):

    with tf.variable_scope('discriminator') as scope:

        if reuse:
            scope.reuse_variables()

        # Convs
        l1 = tf.layers.conv2d(X, filters = 64, kernel_size = [5, 5], strides = [2, 2], padding = 'same',
                              name = 'd_l1_CONV')
        l1 = tf.maximum(l1, LEAK * l1, name = 'd_l1_LEAKYRELU')

        l2 = tf.layers.conv2d(l1, filters = 128, kernel_size = [5, 5], strides = [2, 2], padding = 'same',
                              name = 'd_l2_CONV')
        l2 = tf.maximum(l2, LEAK * l2, name = 'd_l2_LEAKYRELU')

        l3 = tf.layers.conv2d(l2, filters = 256, kernel_size = [5, 5], strides = [2, 2], padding = 'same',
                              name='d_l3_CONV')
        l3 = tf.maximum(l3, LEAK * l3, name = 'd_l3_LEAKYRELU')

        # Reshape
        l4 = tf.contrib.layers.flatten(l3)

        # Ouput
        d_logits = tf.layers.dense(l4, 1, name = 'd_l4_FC')
        d_pred = tf.nn.sigmoid(d_logits, name = 'd_l4_SIGMOID')

        return d_pred, d_logits

## Define loss functions and optimizers ##
X_fake = generator(z)
d_pred_real, logits_real = discriminator(X_real)
d_pred_fake, logits_fake = discriminator(X_fake, reuse = True)
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = tf.ones_like(d_pred_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.zeros_like(d_pred_fake)))
d_loss = d_loss_fake + d_loss_real
tf.summary.scalar('Discriminator loss', d_loss)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(d_pred_fake)))
tf.summary.scalar('Generator loss', g_loss)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]
train_d = tf.train.AdamOptimizer(LR, beta1 = 0.5).minimize(d_loss, var_list = d_vars)
train_g = tf.train.AdamOptimizer(LR, beta1 = 0.5).minimize(g_loss, var_list = g_vars)

## Train network and save model ##
X_batch_tr, Y_batch_tr = extract_tfrecord('../Data/TF_MNIST_train.tfrecords', BATCH_SIZE, IMG_W, IMG_H, IMG_C)
saver = tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(2000):
        img_real = sess.run(X_batch_tr)
        noise = np.random.uniform(-1, 1, size = (BATCH_SIZE, Z_DIM))
        sess.run(train_d, feed_dict={X_real: img_real, z: noise})
        sess.run(train_g, feed_dict={X_real: img_real, z: noise})
        sess.run(train_g, feed_dict={X_real: img_real, z: noise})
        if i % 100 == 0:
            print('Batch num:', i)
            print('Discriminator loss:', sess.run(d_loss, feed_dict = {X_real: img_real, z: noise}))
            print('Generator loss:', sess.run(g_loss, feed_dict={X_real: img_real, z: noise}))
            result = sess.run(merged, feed_dict={X_real: img_real, z: noise})
            writer.add_summary(result, i)

            noise_sam = np.random.uniform(-1, 1, size=(100, Z_DIM))
            pred = sess.run(sampler(z), feed_dict={z: noise_sam})
            img = (np.concatenate([r.reshape(-1, 28)
                                   for r in np.split(pred, 10)
                                   ], axis=-1)).astype(np.uint8)
            Image.fromarray(img).save(
                ('batch_' + str(i)+'.jpg').format(img))
            save_path = saver.save(sess, 'TF_MNIST.ckpt', global_step = i)
    coord.request_stop()
    coord.join(threads)
