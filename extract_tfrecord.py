import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Extract data from the tensorflowRecord format, and creat the input training data for each batch
# Input: tfrecord file dir, batch_size, img_w, img_h, img_c
# Ouput: extracted img data and label data for each batch training
# Example: extract_tfrecord('train.tfrecords', 32, 128, 128, 3)

def extract_tfrecord(tfrecord_name, batch_size, img_w, img_h, img_c):

    filename_queue = tf.train.string_input_producer([tfrecord_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_w, img_h, img_c])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 
    label = tf.cast(features['label'], tf.int32)

    # image = tf.image.per_image_standardization(img)

    # X_batch, Y_batch = tf.train.batch([img, label], batch_size = batch_size, num_threads = 16, capacity = 3000)
    # X_batch, Y_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, num_threads=16,
                                              # capacity=50000, min_after_dequeue=10000)
    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    X_batch, Y_batch = tf.train.shuffle_batch(
        [img, label], batch_size = batch_size, capacity = capacity, num_threads = num_threads,
        min_after_dequeue = min_after_dequeue)

    return X_batch, Y_batch