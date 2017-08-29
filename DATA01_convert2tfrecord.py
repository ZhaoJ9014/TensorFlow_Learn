import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Convert the dataset into the tensorflowRecord format
# Input - dataset list .txt: img_dir label
# Ouput: converted dataset .tfrecords
# Example: convert2tfrecord('./list/list.txt', 128, 128, 'train.tfrecords')

def convert2tfrecord(list_data_set, img_w, img_h, save_name):

    file = open(list_data_set, "r")
    lines = file.readlines()
    file.close()
    N_data = len(lines)
    writer = tf.python_io.TFRecordWriter(save_name)
    for i in range(N_data):
        label = int(lines[i].split()[1])
        img = Image.open(lines[i].split()[0])
        img = img.resize((img_w, img_h))
        img_raw = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())

    writer.close()
