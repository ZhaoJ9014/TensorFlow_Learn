import tensorflow as tf
from PIL import Image
import os

# Conduct offline data augmentation, including flip_up_down, flip_left_right, random_brightness,
# random_contrast, random_saturation, and random_crop
# Input: data_set_name, list_data_set, img_w, img_h, is_flip_up_down_true, is_flip_left_right_true,
# is_random_brightness_true, is_random_contrast_true, is_random_saturation_true, is_random_crop_true
# Ouput: corresponding augmented dataset
# Example: data_augmentation('toy_data', './list/list.txt', img_w=128, img_h=128, is_flip_up_down_true=False,
# is_flip_left_right_true=False, is_random_brightness_true=False,
# is_random_contrast_true=False, is_random_saturation_true=False, is_random_crop_true=True)

def data_augmentation(data_set_name, list_data_set, img_w=128, img_h=128, is_flip_up_down_true=False,
                      is_flip_left_right_true=False, is_random_brightness_true=False,
                      is_random_contrast_true=False, is_random_saturation_true=False, is_random_crop_true=False):

    file = open(list_data_set, "r")
    lines = file.readlines()
    file.close()
    N_data = len(lines)

    if is_flip_up_down_true:
        try:
            os.mkdir(data_set_name + '_flip_up_down')
        except Exception:
            pass
        for i in range(N_data):
            raw_image = Image.open('./' + data_set_name + '/' +  lines[i].split()[0])
            raw_image = raw_image.resize((img_w, img_h))
            xs = tf.placeholder("uint8", [img_w, img_h, 3])
            img = tf.image.flip_up_down(xs)
            with tf.Session() as session:
                result = session.run(img, feed_dict={xs: raw_image})
            img = Image.fromarray(result, 'RGB')
            try:
                os.mkdir(data_set_name + '_flip_up_down/' + lines[i].split()[0].split('/')[0])
            except Exception:
                pass
            img.save('./' + data_set_name + '_flip_up_down/' + lines[i].split()[0])

    if is_flip_left_right_true:
        try:
            os.mkdir(data_set_name + '_flip_left_right')
        except Exception:
            pass
        for i in range(N_data):
            raw_image = Image.open('./' + data_set_name + '/' + lines[i].split()[0])
            raw_image = raw_image.resize((img_w, img_h))
            xs = tf.placeholder("uint8", [img_w, img_h, 3])
            img = tf.image.flip_left_right(xs)
            with tf.Session() as session:
                result = session.run(img, feed_dict={xs: raw_image})
            img = Image.fromarray(result, 'RGB')
            try:
                os.mkdir(data_set_name + '_flip_left_right/' + lines[i].split()[0].split('/')[0])
            except Exception:
                pass
            img.save('./' + data_set_name + '_flip_left_right/' + lines[i].split()[0])

    if is_random_brightness_true:
        try:
            os.mkdir(data_set_name + '_random_brightness')
        except Exception:
            pass
        for i in range(N_data):
            raw_image = Image.open('./' + data_set_name + '/' + lines[i].split()[0])
            raw_image = raw_image.resize((img_w, img_h))
            xs = tf.placeholder("uint8", [img_w, img_h, 3])
            img = tf.image.random_brightness(xs, max_delta=0.3)
            with tf.Session() as session:
                result = session.run(img, feed_dict={xs: raw_image})
            img = Image.fromarray(result, 'RGB')
            try:
                os.mkdir(data_set_name + '_random_brightness/' + lines[i].split()[0].split('/')[0])
            except Exception:
                pass
            img.save('./' + data_set_name + '_random_brightness/' + lines[i].split()[0])

    if is_random_contrast_true:
        try:
            os.mkdir(data_set_name + '_random_contrast')
        except Exception:
            pass
        for i in range(N_data):
            raw_image = Image.open('./' + data_set_name + '/' + lines[i].split()[0])
            raw_image = raw_image.resize((img_w, img_h))
            xs = tf.placeholder("uint8", [img_w, img_h, 3])
            img = tf.image.random_contrast(xs, 0.8, 1.2)
            with tf.Session() as session:
                result = session.run(img, feed_dict={xs: raw_image})
            img = Image.fromarray(result, 'RGB')
            try:
                os.mkdir(data_set_name + '_random_contrast/' + lines[i].split()[0].split('/')[0])
            except Exception:
                pass
            img.save('./' + data_set_name + '_random_contrast/' + lines[i].split()[0])

    if is_random_saturation_true:
        try:
            os.mkdir(data_set_name + '_random_saturation')
        except Exception:
            pass
        for i in range(N_data):
            raw_image = Image.open('./' + data_set_name + '/' + lines[i].split()[0])
            raw_image = raw_image.resize((img_w, img_h))
            xs = tf.placeholder("uint8", [img_w, img_h, 3])
            img = tf.image.random_saturation(xs, 0.3, 0.5)
            with tf.Session() as session:
                result = session.run(img, feed_dict={xs: raw_image})
            img = Image.fromarray(result, 'RGB')
            try:
                os.mkdir(data_set_name + '_random_saturation/' + lines[i].split()[0].split('/')[0])
            except Exception:
                pass
            img.save('./' + data_set_name + '_random_saturation/' + lines[i].split()[0])

    if is_random_crop_true:
        try:
            os.mkdir(data_set_name + '_random_crop')
        except Exception:
            pass
        for i in range(N_data):
            raw_image = Image.open('./' + data_set_name + '/' + lines[i].split()[0])
            raw_image = raw_image.resize((img_w, img_h))
            xs = tf.placeholder("uint8", [img_w, img_h, 3])
            img = tf.random_crop(xs, [128, 128, 3])
            with tf.Session() as session:
                result = session.run(img, feed_dict={xs: raw_image})
            img = Image.fromarray(result, 'RGB')
            try:
                os.mkdir(data_set_name + '_random_crop/' + lines[i].split()[0].split('/')[0])
            except Exception:
                pass
            img.save('./' + data_set_name + '_random_crop/' + lines[i].split()[0])

    return None