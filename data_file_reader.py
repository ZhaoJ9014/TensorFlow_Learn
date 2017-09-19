import random

# Read data file list (img dir & label), and conduct shuffle (if true)
# Input - file_dir, is_shuffle
# Ouput: extracted file lines, including img dir & labels
# Example: data_file_reader(file_dir = '/home/zhaojian/Documents/Projects/TF_MNIST/List/train.txt', is_shuffle = True)

def data_file_reader(file_dir, is_shuffle = False):

    file = open(file_dir, "r")
    lines = file.readlines()
    file.close()

    if is_shuffle:

        random.shuffle(lines)

    return lines

# file_content = data_file_reader(file_dir = '/home/zhaojian/Documents/Projects/TF_MNIST/List/train.txt', is_shuffle = True)















