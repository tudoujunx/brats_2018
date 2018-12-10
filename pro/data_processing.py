import tensorflow as tf
import numpy as np

train_load_dir = 'D:/program/pro_code/brats_2018/data_set/pre_data/output_train.tfrecords'
pretrain_data_1 = 'D:/program/pro_code/brats_2018/data_set/pre_data/output_train_pre1.tfrecords'

image_size = 240
slices = 155
BATCH_SIZE = 1

def read_img(train_image_filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, image_example = reader.read(train_image_filename_queue)
    features = tf.parse_single_example(
        image_example,
        features={
            'name': tf.FixedLenFeature([], tf.string),
            'image_type': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'flair': tf.FixedLenFeature([], tf.string),
            'T1': tf.FixedLenFeature([], tf.string),
            'T1ce': tf.FixedLenFeature([], tf.string),
            'T2': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    flair_img, t1_img, t1ce_img, t2_img = features['flair'], features['T1'], features['T1ce'], features['T2']

    label = tf.reshape(tf.decode_raw(label, tf.int32), [image_size, image_size, slices])
    flair_img = tf.reshape(tf.decode_raw(flair_img, tf.int32), [image_size, image_size, slices])
    t1_img = tf.reshape(tf.decode_raw(t1_img, tf.int32), [image_size, image_size, slices])
    t1ce_img = tf.reshape(tf.decode_raw(t1ce_img, tf.int32), [image_size, image_size, slices])
    t2_img = tf.reshape(tf.decode_raw(t2_img, tf.int32), [image_size, image_size, slices])

    image_data = [flair_img, t1_img, t1ce_img, t2_img]
    image_data = tf.transpose(image_data, [1, 2, 0, 3])

    min_after_dequeue = 3

    capacity = min_after_dequeue + 3 * batch_size

    image_batch, label_batch = tf.train.shuffle_batch(
        [image_data, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch

