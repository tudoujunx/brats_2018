import os
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt


DIR = 'D:/program/brats-2018/MICCAI_BraTS_2018_Data_Training'
train_load_dir = 'D:/program/python/test/BratsData/predata/output_train.tfrecords'
cv_load_dir =  'D:/program/python/test/BratsData/predata/output_cv.tfrecords'

image_size = 240
channels = 155

files = tf.train.match_filenames_once(train_load_dir)
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, image_example = reader.read(filename_queue)
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
name, image_type = features['name'], features['image_type']

label = features['label']
label = tf.decode_raw(label, tf.uint8)
print(label)
#label.reshape([image_size, image_size])
np.reshape(label, [image_size, image_size])

flair_img, t1_img, t1ce_img, t2_img = features['flair'], features['T1'], features['T1ce'], features['T2']

flair_img = tf.decode_raw(flair_img, tf.uint16)
t1_img = tf.decode_raw(t1_img, tf.uint16)
t1ce_img = tf.decode_raw(t1ce_img, tf.uint16)
t2_img = tf.decode_raw(t2_img, tf.uint16)

flair_img.set_shape([image_size, image_size, channels])
t1_img.set_shape([image_size, image_size, channels])
t1ce_img.set_shape([image_size, image_size, channels])
t2_img.set_shape([image_size, image_size, channels])

image_Data = [flair_img, t1_img, t1ce_img, t2_img]
"""
image_Data = [image_size, image_size, channels, 4]
for i in range(channels):
    image_Data[:, :, i, 0] = flair_img[:, :, i]
    image_Data[:, :, i, 1] = t1_img[:, :, i]
    image_Data[:, :, i, 2] = t1ce_img[:, :, i]
    image_Data[:, :, i, 3] = t2_img[:, :, i]
"""
#预留的预处理程序

min_after_dequeue = 30
batch_size = 3
learning_rate = 0.01
capacity = min_after_dequeue + 3*batch_size

image_batch, label_batch = tf.train.shuffle_batch(
    [image_Data, label], batch_size=batch_size,
    capacity=capacity, min_after_dequeue=min_after_dequeue)
"""
logit = inference(image_batch)#训练网络
loss = calc_loss(image_batch, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
"""
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    TRAINING_ROUNDS = 1000
    for i in range(TRAINING_ROUNDS):
        sess.run(image_batch)

    coord.request_stop()
    coord.join(threads)






"""
#f = open('image/Brats18_2013_2_1_t1.nii')
img = nib.load('D:/program/brats-2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1.nii.gz')
print(img)
img_data = img.get_fdata()
a = img_data[:, :, 80]
plt.imshow(img_data[:, :, 80], cmap="gray", origin="lower")
plt.show()


img_data = img.get_fdata()
print(img_data.shape)
slice_0 = img_data[100, :, :]
slice_1 = img_data[:, 100, :]
slice_2 = img_data[:, :, 100]

plt.suptitle("Some slices of the image")
plt.imshow(img_data[:, :, 80], cmap="gray", origin="lower")
plt.show()


def slice_show(di,asex):
    for j in range(len(img_data[:, 0, 0])):
        asex.imshow(img_data[j, :, :],cmap="gary")
        plt.show()
        time.sleep(0.5)

for i in [0, 1, 2]:
    asex = plt.subplot(2, 2, i+1)
    slice_show(i, asex)"""



