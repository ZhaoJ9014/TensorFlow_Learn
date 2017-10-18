import tensorflow as tf
import numpy as np

from data_file_reader import data_file_reader
from data_generator import data_generator

## Data file ##
MNIST_file = data_file_reader(file_dir = '/home/zhaojian/Documents/Projects/TF_MNIST/List/train.txt',
                              is_shuffle = True)

## Paramaters ##
IMG_W = 28
IMG_H = 28
IMG_C = 1
NUM_CLS = 10

## Weight initialization ##
def xavier_init(num_in, num_out, constant = 1.0):

    low = -constant * np.sqrt(6.0 / (num_in + num_out))
    high = constant * np.sqrt(6.0 / (num_in + num_out))

    return tf.random_uniform((num_in, num_out), minval = low, maxval = high, dtype = tf.float32)

# ## Placeholders ##
# x = tf.placeholder(dtype=tf.float32, shape=[None, IMG_W, IMG_H, IMG_C], name='x_in')
# y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_in')
#
# ## Network ##
# reshape1 = tf.reshape(x, shape=[-1, IMG_W*IMG_H*IMG_C], name='reshape1')
#
# W_fc1 = tf.Variable(xavier_init(num_in=IMG_W*IMG_H*IMG_C, num_out=1024), name='W_fc1')
# b_fc1 = tf.Variable(tf.zeros([1024]), dtype = tf.float32)
# mul_fc1 = tf.matmul(reshape1, W_fc1, name='mul_fc1')
# add_fc1 = tf.add(mul_fc1, b_fc1, name='add_fc1')
# fc1 = tf.nn.relu(add_fc1, name='fc1')
#
# W_fc2 = tf.Variable(xavier_init(num_in=1024, num_out=10), name='W_fc2')
# b_fc2 = tf.Variable(tf.zeros([10]), dtype = tf.float32)
# mul_fc2 = tf.matmul(fc1, W_fc2)
# fc2 = tf.add(mul_fc2, b_fc2, name='fc2')
#
# predictions = tf.nn.softmax(fc2, name = 'softmax_tensor')
#
# onehot_labels = tf.reshape(tf.one_hot(indices=tf.cast(y, tf.int32), depth=10), [-1, 10])
#
# loss = tf.losses.softmax_cross_entropy(
#       onehot_labels = onehot_labels,
#       logits = fc2)
#
# correct_prediction = tf.equal(tf.argmax(onehot_labels, 1), tf.argmax(predictions, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
# train_op = optimizer.minimize(
#       loss = loss)
#
# init = tf.global_variables_initializer()
#
# ## Train ##
# with tf.Session() as sess:
#
#     sess.run(init)
#
#     for _ in range(int(round(len(MNIST_file) / 128))):
#         X, Y = data_generator('/home/zhaojian/Documents/Projects/TF_MNIST/Data/train/',
#                               file_content = MNIST_file, img_w = 28, img_h = 28, img_c = 1, batch_size = 128,
#                               batch_index = _)
#         sess.run(train_op, feed_dict={x: X, y: Y})
#         print sess.run(accuracy, feed_dict={x: X, y: Y})
#         if _ % 100 == 0:
#             tr_loss  = sess.run(loss, feed_dict={x: X, y: Y})
#             print 'Batch num:', _, 'Training loss:', tr_loss
#     # Save model
#     saver = tf.train.Saver()
#     saver.save(sess, './model/my_test_model')

## Finetuning ##
with tf.Session() as sess:
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('./model/my_test_model.meta')
    graph = tf.get_default_graph()
    # Get input tensors and bottleneck layer
    new_x = graph.get_tensor_by_name("x_in:0")
    new_y = graph.get_tensor_by_name("y_in:0")
    new_fc1 = graph.get_tensor_by_name("fc1:0")
    # Add new layer
    new_W_fc2 = tf.Variable(xavier_init(1024, 100))
    new_b_fc2 = tf.Variable(tf.constant(0.05, shape=[100]))
    new_fc2 = tf.matmul(new_fc1, new_W_fc2)
    new_fc2 = tf.add(new_fc2, new_b_fc2, name='new_fc2')
    # Initialize new variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Load pre-trained weights
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))


    new_predictions = tf.nn.softmax(new_fc2, name = 'new_softmax_tensor')

    new_onehot_labels = tf.reshape(tf.one_hot(indices=tf.cast(new_y, tf.int32), depth=100), [-1, 100])

    new_loss = tf.losses.softmax_cross_entropy(onehot_labels = new_onehot_labels,
                                               logits = new_fc2)

    new_correct_prediction = tf.equal(tf.argmax(new_onehot_labels, 1), tf.argmax(new_predictions, 1))
    new_accuracy = tf.reduce_mean(tf.cast(new_correct_prediction, tf.float32))

    new_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    new_train_op = new_optimizer.minimize(loss = new_loss)


    for _ in range(int(round(len(MNIST_file) / 32))):
        X, Y = data_generator('/home/zhaojian/Documents/Projects/TF_MNIST/Data/train/',
                              file_content = MNIST_file, img_w = 28, img_h = 28, img_c = 1, batch_size = 32,
                              batch_index = _)
        sess.run(new_train_op, feed_dict={new_x: X, new_y: Y})
        if _ % 10 == 0:
            new_tr_loss  = sess.run(new_loss, feed_dict={new_x: X, new_y: Y})
            print 'Batch num:', _, 'Training loss:', new_tr_loss
    # Save model
    saver = tf.train.Saver()
    saver.save(sess, './model/my_finetune_model')
