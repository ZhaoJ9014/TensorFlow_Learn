import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

## Weight initialization ##
def xavier_init(num_in, num_out, constant = 1.0):

    low = -constant * np.sqrt(6.0 / (num_in + num_out))
    high = constant * np.sqrt(6.0 / (num_in + num_out))

    return tf.random_uniform((num_in, num_out), minval = low, maxval = high, dtype = tf.float32)


## AutoEncoder class ##
class AdditiveGaussianNoiseAutoencoder(object):

    def __init__(self, num_in, num_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):

        self.num_in = num_in
        self.num_hidden = num_hidden
        self.transfer = transfer_function
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.scale = tf.placeholder(dtype=tf.float32)
        self.x = tf.placeholder(dtype = tf.float32, shape = [None, self.num_in])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + self.scale * tf.random_normal([self.num_in]),
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):

        all_weights = dict()

        all_weights['w1'] = tf.Variable(xavier_init(self.num_in, self.num_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.num_hidden]), dtype = tf.float32)

        all_weights['w2'] = tf.Variable(tf.zeros([self.num_hidden, self.num_in], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.num_in]), dtype=tf.float32)

        return all_weights

    def partial_fit(self, X):

        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: 0.01})

        return cost

    def calc_total_cost(self, X):

        return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: 0.01})

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

## Pre-processor ##
def standard_scale(X_train, X_test):

    prepocessor = prep.StandardScaler().fit(X_train)
    X_train = prepocessor.transform(X_train)
    X_test = prepocessor.transform(X_test)

    return X_train, X_test

## Batch iterater ##
def get_random_batch_from_data(data, batch_size):

    start_index = np.random.randint(0, len(data) - batch_size)

    return data[start_index:(start_index + batch_size)]


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

num_samples = int(mnist.train.num_examples)
num_epoches = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(num_in=784, num_hidden=200, transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

for epoch in range(num_epoches):

    avg_cost = 0.
    total_batch = int(num_samples / batch_size)

    for i in range(total_batch):

        batch_X = get_random_batch_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_X)

        avg_cost += cost / num_samples * batch_size

    if epoch % display_step == 0:

        print('Epoch:', '%04d' % (epoch + 1),
              'Cost:', '{:.9f}'.format(avg_cost))

    print('Total cost:' + str(autoencoder.calc_total_cost(X_test)))
