import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST', one_hot = False)

# Hyperparameters ##
batch_size = 500
## Placeholders ##
X_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28 * 28])
Y_in = tf.placeholder(dtype=tf.float32, shape=[batch_size])
## Main Network ##
reshape0 = tf.reshape(X_in, shape=[batch_size, 28, 28, 1])

conv1 = tf.layers.conv2d(
      inputs = reshape0,
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
tf.summary.scalar('loss', loss)

correct_prediction = tf.equal(tf.argmax(onehot_labels, 1), tf.argmax(predictions, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', accuracy)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
train_op = optimizer.minimize(
      loss = loss)

## Train network and save model ##
saver = tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    i = 0
    for i in range(1000):
        for dev_ind in range(2):
            with tf.device('/gpu: %d' % dev_ind):
                img_tr, label_tr = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={X_in: img_tr, Y_in: label_tr})
                if i % 200 == 0:
                    print('batch num:', i)
                    print('loss:', sess.run(loss, feed_dict={X_in: img_tr, Y_in: label_tr}))
                    print('acc:', sess.run(accuracy, feed_dict={X_in: img_tr, Y_in: label_tr}))
                    result = sess.run(merged, feed_dict={X_in: img_tr, Y_in: label_tr})
                    writer.add_summary(result, i)
            save_path = saver.save(sess, 'TF_MNIST.ckpt', global_step = i)
