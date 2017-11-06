import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

## Training data
mnist = input_data.read_data_sets('./tmp/mnist/',one_hot=True)

## Define hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

## Define network parameters
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

## Define placeholders
x = tf.placeholder(dtype=tf.float32,shape=[None,n_steps,n_inputs])
y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes])

## Define weights
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1,shape=[n_hidden_units])),
    'out': tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

## Define RNN
def RNN(X, weights, biases):
    X = tf.reshape(X,shape=[-1,n_inputs])
    X = tf.matmul(X,weights['in'])+biases['in']
    X = tf.reshape(X,shape=[-1,n_steps,n_hidden_units])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,X,initial_state=init_state,time_major=False)
    results = tf.matmul(final_state[1],weights['out'])+biases['out']
    return results

pred = RNN(x,weights,biases)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size,n_steps,n_inputs])
        sess.run(train_op,feed_dict={x:batch_x,y:batch_y})
        if step%20 == 0:
            print(sess.run(accuracy,feed_dict={x:batch_x,y:batch_y}))
        step = step + 1
