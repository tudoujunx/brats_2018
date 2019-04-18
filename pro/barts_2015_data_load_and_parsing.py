import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
# import csv

train_data_dir = r'..\data_set\pre_data\brats_2015\training'
train_load_dir = '..\data_set\pre_data\output_train_2015.tfrecords'
cv_load_dir = '..\data_set\pre_data\output_cv_2015.tfrecords'
test_data_dir = r'..\data_set\pre_data\brats_2015\testing'
test_load_dir = '..\data_set\pre_data\output_test_2015.tfrecords'



img_width, img_height, slices = 240, 240, 155
min_after_dequeue = 15

""""""
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _encode_image(root, files):
    if len(files) == 5:
        for file in files:
            img = sitk.ReadImage(os.path.join(root, file))
            img_data = sitk.GetArrayFromImage(img)
            img_data = img_data.astype(int)
            if '.OT.' in file:
                label = img_data.tobytes()
            if '_Flair.' in file:
                flair_img = img_data.tobytes()
            if '_T1.' in file:
                t1_img = img_data.tobytes()
            if '_T1c.' in file:
                t1ce_img = img_data.tobytes()
            if '_T2.' in file:
                t2_img = img_data.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(label),
            'flair': _bytes_feature(flair_img),
            'T1': _bytes_feature(t1_img),
            'T1ce': _bytes_feature(t1ce_img),
            'T2': _bytes_feature(t2_img)
        }))

    elif len(files) == 4:
        for file in files:
            img = sitk.ReadImage(os.path.join(root, file))
            img_data = sitk.GetArrayFromImage(img)
            img_data = img_data.astype(int)
            if '_Flair.' in file:
                flair_img = img_data.tobytes()
            if '_T1.' in file:
                t1_img = img_data.tobytes()
            if '_T1c.' in file:
                t1ce_img = img_data.tobytes()
            if '_T2.' in file:
                t2_img = img_data.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'flair': _bytes_feature(flair_img),
            'T1': _bytes_feature(t1_img),
            'T1ce': _bytes_feature(t1ce_img),
            'T2': _bytes_feature(t2_img)
        }))
    return example



def train_data_load():
    """
    Load image data as .tfrecord format
    """
    writer_train = tf.python_io.TFRecordWriter(train_load_dir)
    writer_cv = tf.python_io.TFRecordWriter(cv_load_dir)

    for root, dirs, files in os.walk(train_data_dir):
        print(root, dirs, files) #当前目录路径

        if len(files) == 5:

            example = _encode_image(root, files)

            chance = np.random.randint(100)
            if chance < 10:
                writer_cv.write(example.SerializeToString())
            else:
                writer_train.write(example.SerializeToString())

        print('----------------------------------')
    writer_cv.close()
    writer_train.close()
    print('Load success')


def test_data_load():
    """
       Load image data as .tfrecord format
       """
    writer_test = tf.python_io.TFRecordWriter(test_load_dir)

    for root, dirs, files in os.walk(test_data_dir):
        print(root, dirs, files)  # 当前目录路径

        if len(files) == 4:

            example = _encode_image(root, files)

            writer_test.write(example.SerializeToString())

        print('----------------------------------')
    writer_test.close()
    print('Load success')


def parsing_train_data(image_filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, image_example = reader.read(image_filename_queue)
    features = tf.parse_single_example(
        image_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'flair': tf.FixedLenFeature([], tf.string),
            'T1': tf.FixedLenFeature([], tf.string),
            'T1ce': tf.FixedLenFeature([], tf.string),
            'T2': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    flair_img, t1_img, t1ce_img, t2_img = features['flair'], features['T1'], features['T1ce'], features['T2']

    label = tf.reshape(tf.decode_raw(label, tf.int32), [img_width, img_height, slices])
    flair_img = tf.reshape(tf.decode_raw(flair_img, tf.int32), [img_width, img_height, slices])
    t1_img = tf.reshape(tf.decode_raw(t1_img, tf.int32), [img_width, img_height, slices])
    t1ce_img = tf.reshape(tf.decode_raw(t1ce_img, tf.int32), [img_width, img_height, slices])
    t2_img = tf.reshape(tf.decode_raw(t2_img, tf.int32), [img_width, img_height, slices])

    image_data = [flair_img, t1_img, t1ce_img, t2_img]
    image_data = tf.transpose(image_data, [1, 2, 0, 3])

    capacity = min_after_dequeue + 3 * batch_size

    image_batch, label_batch = tf.train.shuffle_batch(
        [image_data, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


def parsing_test_data(image_filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, image_example = reader.read(image_filename_queue)
    features = tf.parse_single_example(
        image_example,
        features={
            'flair': tf.FixedLenFeature([], tf.string),
            'T1': tf.FixedLenFeature([], tf.string),
            'T1ce': tf.FixedLenFeature([], tf.string),
            'T2': tf.FixedLenFeature([], tf.string)
        }
    )
    flair_img, t1_img, t1ce_img, t2_img = features['flair'], features['T1'], features['T1ce'], features['T2']

    flair_img = tf.reshape(tf.decode_raw(flair_img, tf.int32), [img_width, img_height, slices])
    t1_img = tf.reshape(tf.decode_raw(t1_img, tf.int32), [img_width, img_height, slices])
    t1ce_img = tf.reshape(tf.decode_raw(t1ce_img, tf.int32), [img_width, img_height, slices])
    t2_img = tf.reshape(tf.decode_raw(t2_img, tf.int32), [img_width, img_height, slices])

    image_data = [flair_img, t1_img, t1ce_img, t2_img]
    image_data = tf.transpose(image_data, [1, 2, 0, 3])

    capacity = min_after_dequeue + 3 * batch_size

    image_batch = tf.train.shuffle_batch(
        [image_data], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    return image_batch


def test():
    pass


if __name__ == '__main__':
    test()
