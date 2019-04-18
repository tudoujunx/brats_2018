import os
import nibabel as nib
import numpy as np
import tensorflow as tf
# import csv
from albumentations import *
import cv2 as cv
import SimpleITK as sitk

data_dir = '../data_set/pre_data/MICCAI_BraTS_2018_Data_Training'
train_load_dir = '../data_set/pre_data/output_train.tfrecords'
cv_load_dir = '../data_set/pre_data/output_cv.tfrecords'
test_data_dir = '../data_set/pre_data/MICCAI_BraTS_2018_Data_Validation'
test_load_dir = '../data_set/pre_data/output_test.tfrecords'

# image_save_dir = r'..\data_set\pre_data\image'

img_width, img_height, channels, slices = 240, 240, 4, 155
min_after_dequeue = 15


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _encode_image(root, files):

    if len(files) == 5:
        for file in files:
            img = nib.load(os.path.join(root, file))
            img_data = img.get_fdata()
            img_data = img_data.astype(int)
            if '_seg' in file:
                img_data = np.where(img_data < 3, img_data, 3)
                label = img_data.tobytes()
            if '_flair' in file:
                flair_img = img_data
            if '_t1.' in file:
                t1_img = img_data
            if '_t1ce' in file:
                t1ce_img = img_data
            if '_t2' in file:
                # (filename, extension) = os.path.splitext(file)
                # patient_name = filename.split('.')[0].strip('t2')
                t2_img = img_data

        image = [flair_img, t1_img, t1ce_img, t2_img]
        image = np.transpose(image, (1, 2, 0, 3))
        image = _to_bytes(image)

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(label),
            'image': _bytes_feature(image),
        }))

    return example


def img_aug(image, mask):
    aug = Compose([
        OneOf([RandomSizedCrop(min_max_height=(120, 240), height=240, width=240, p=0.5),
               PadIfNeeded(min_height=240, min_width=240, p=0.5)], p=1),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8)])

    augmented = aug(image=image, mask=mask)

    image_heavy = augmented['image']
    mask_heavy = augmented['mask']

    return image_heavy, mask_heavy


def _to_bytes(img_data):
    max = img_data.max()
    min = img_data.min()
    img_data = 255*(img_data - min)/(max - min)
    img_data = img_data.astype('uint8')

    # for i in range(slices):
    # cv.imwrite(os.path.join(image_save_dir, name + str(i) + '.png'), img_data[:, :, :, i])

    return img_data.tobytes()


def train_data_load():
    """
    Load image data as .tfrecord format
    """
    writer_train = tf.python_io.TFRecordWriter(train_load_dir)
    writer_cv = tf.python_io.TFRecordWriter(cv_load_dir)

    for root, dirs, files in os.walk(data_dir):
        print(root, dirs, files)  # 当前目录路径

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

        if len(files):
            if len(files) == 1:
                """
                age_file = csv.reader(open(root + '/' + files[0], 'r'))
                age_file = list(age_file)
                for i in range(0, len(age_file)):
                    age_file[i] = str(age_file[i])
                age_file = ','.join(age_file)
                age_file = bytes(age_file, "utf8")
                example = tf.train.Example(features=tf.train.Features(feature={
                    'type': _bytes_feature(b'age'),
                    'agePre': _bytes_feature(age_file)
                }))
                writer.write(example.SerializeToString())"""

            elif len(files) == 4:
                for file in files:
                    print(file)
                    (filename, extension) = os.path.splitext(file)

                    img = nib.load(root + '/' + file)
                    img_data = img.get_fdata()
                    # img_data = img_data.tolist()
                    # img_max = img_data.max()
                    # img_min = img_data.min()
                    # img_data = (img_data-(img_max/2))/(img_max-img_min)
                    # img_data = img_data.astype('float32')

                    img_data = img_data.astype(int)

                    if '_flair' in filename:
                        flair_img = img_data.tobytes()

                    if '_t1' in filename:
                        t1_img = img_data.tobytes()

                    if '_t1ce' in filename:
                        t1ce_img = img_data.tobytes()

                    if '_t2' in filename:
                        t2_img = img_data.tobytes()
                        patient_name = bytes(filename.split('.')[0].strip('_t2'), "utf8")

                example = tf.train.Example(features=tf.train.Features(feature={
                    'name': _bytes_feature(patient_name),
                    'flair': _bytes_feature(flair_img),
                    'T1': _bytes_feature(t1_img),
                    'T1ce': _bytes_feature(t1ce_img),
                    'T2': _bytes_feature(t2_img)
                }))

                writer_test.write(example.SerializeToString())

                patient_name = None
                flair_img = None
                t1_img = None
                t1ce_img = None
                t2_img = None

            print('----------------------------------')
    writer_test.close()
    print('Load success')


def parsing_data(image_filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, image_example = reader.read(image_filename_queue)
    features = tf.parse_single_example(
        image_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        }
    )
    label = features['label']
    image = features['image']
    label = tf.reshape(tf.decode_raw(label, tf.int32), [img_width, img_height, slices])
    image = tf.reshape(tf.decode_raw(image, tf.uint8), [img_width, img_height, channels, slices])

    capacity = min_after_dequeue + 3 * batch_size

    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


def parsing_test_data(image_filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, image_example = reader.read(image_filename_queue)
    features = tf.parse_single_example(
        image_example,
        features={
            'name': tf.FixedLenFeature([], tf.string),
            'flair': tf.FixedLenFeature([], tf.string),
            'T1': tf.FixedLenFeature([], tf.string),
            'T1ce': tf.FixedLenFeature([], tf.string),
            'T2': tf.FixedLenFeature([], tf.string)
        }
    )
    name = features['name']
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
    return image_batch, name


if __name__ == '__main__':
    # train_data_load()
    pass
