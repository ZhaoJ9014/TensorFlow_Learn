import numpy as np
from PIL import Image
from data_file_reader import data_file_reader

# Generate training/testing data
# Input - data_source_route, file_content, img_w, img_h, img_c, batch_size, batch_index
# Ouput: training data per batch, or testing data
# Example: data_generator('/home/zhaojian/Documents/Projects/TF_MNIST/Data/train/',
# file_content = file_content, img_w = 28, img_h = 28, img_c = 1, batch_size = 128, batch_index = 0)

def data_generator(data_source_route, file_content, img_w, img_h, img_c, batch_size, batch_index = 0):

    N = len(file_content)
    X_batch = np.zeros((batch_size, img_w, img_h, img_c), dtype = np.float32)
    Y_batch = np.zeros((batch_size, 1), dtype = np.int32)
    file_list_batch = file_content[batch_index * batch_size : (batch_index + 1) * batch_size]

    for i in range(batch_size):

        Y_batch[i] = int(file_list_batch[i].split()[1])
        tmp = Image.open(data_source_route + file_list_batch[i].split()[0])
        tmp = tmp.resize((img_w, img_h))
        tmp = tmp / np.float32(128) - np.float32(1)
        X_batch[i, ...] = np.expand_dims(tmp, -1)

    return X_batch, Y_batch

# file_content = data_file_reader(file_dir = '/home/zhaojian/Documents/Projects/TF_MNIST/List/train.txt', is_shuffle = True)
# X_batch, Y_batch = data_generator(data_source_route = '/home/zhaojian/Documents/Projects/TF_MNIST/Data/train/',
#                                   file_content = file_content, img_w = 28, img_h = 28, img_c = 1, batch_size = 128, batch_index = 0)
# print X_batch.shape


















