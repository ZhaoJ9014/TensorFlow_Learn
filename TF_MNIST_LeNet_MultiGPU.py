import tensorflow as tf
from data_file_reader import data_file_reader
from data_generator import data_generator

# Hyperparameters ##
batch_size = 128
## Placeholders ##
X_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1])
Y_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])
## Main Network ##

conv1 = tf.layers.conv2d(
      inputs = X_in,
      filters = 32,
      kernel_size = [5, 5],
      padding = "same",
      activation = tf.nn.relu)

pool1 = tf.layers.max_pooling2d(
      inputs = conv1,
      pool_size = [2, 2],
      strides = 2)

conv2 = tf.layers.conv2d(
      inputs = pool1,
      filters = 64,
      kernel_size = [5, 5],
      padding = "same",
      activation = tf.nn.relu)

pool2 = tf.layers.max_pooling2d(
      inputs = conv2,
      pool_size = [2, 2],
      strides = 2)

pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

dense = tf.layers.dense(
      inputs = pool2_flat,
      units = 1024,
      activation = tf.nn.relu)

dropout = tf.layers.dropout(
      inputs = dense,
      rate = 0.4)

logits = tf.layers.dense(
      inputs = dropout,
      units = 10)

predictions = tf.nn.softmax(logits, name = 'softmax_tensor')

onehot_labels = tf.reshape(tf.one_hot(indices=tf.cast(Y_in, tf.int32), depth=10), [-1, 10])

loss = tf.losses.softmax_cross_entropy(
      onehot_labels = onehot_labels,
      logits = logits)

correct_prediction = tf.equal(tf.argmax(onehot_labels, 1), tf.argmax(predictions, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
train_op = optimizer.minimize(
      loss = loss)

file_content = data_file_reader(file_dir = '/home/zhaojian/Documents/Projects/TF_MNIST/List/train.txt', is_shuffle = True)
## Train network and save model ##
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):

        for ind in range(2):
            with tf.device('/gpu: %d' % ind):

                img_tr, label_tr = data_generator(data_source_route = '/home/zhaojian/Documents/Projects/TF_MNIST/Data/train/',
                                                  file_content = file_content, img_w = 28, img_h = 28, img_c = 1, batch_size = 128, batch_index = 0)
                sess.run(train_op, feed_dict={X_in: img_tr, Y_in: label_tr})
                if i % 200 == 0:
                    print('batch num:', i)
                    print('loss:', sess.run(loss, feed_dict={X_in: img_tr, Y_in: label_tr}))
                    print('acc:', sess.run(accuracy, feed_dict={X_in: img_tr, Y_in: label_tr}))
