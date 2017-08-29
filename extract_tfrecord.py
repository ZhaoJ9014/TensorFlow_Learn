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

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [img_w, img_h, img_c])
    label = tf.cast(features['label'], tf.int32)

    # X_batch, Y_batch = tf.train.batch([image, label], batch_size = batch_size, num_threads = 16, capacity = 3000)
    X_batch, Y_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=16,
                                              capacity=50000, min_after_dequeue=10000)

    return X_batch, Y_batch

    # # return image, label
    # with tf.Session() as sess:
    #     init_op = tf.initialize_all_variables()
    #     sess.run(init_op)
    #     coord=tf.train.Coordinator()
    #     threads= tf.train.start_queue_runners(coord=coord)
    #     for i in range(5):
    #         example, l = sess.run([image,label])
    #         img=Image.fromarray(example, 'RGB')
    #         img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')
    #         print(example, l)
    #     coord.request_stop()
    #     coord.join(threads)