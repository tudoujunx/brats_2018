import tensorflow as tf
import numpy as np
import os

dir = "D:/program/pro_code/brats_2018/data_set/pre_data/output_train.tfrecords"

image_size = 240
slices = 155

def read_img(train_image_filename_queue):
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

    return image_data, label

def main():
    train_file_path = os.path.join(dir)
    train_image_filename_queue = tf.train.string_input_producer(
        string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True
    )
    _, label = read_img(train_image_filename_queue)
    weight = np.zeros([5])
    w = np.zeros([5])
    flag = 0

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                labels = sess.run(label)
                flag += 1
                for i in range(image_size):
                    for j in range(image_size):
                        for k in range(slices):
                            weight[labels[i, j, k]] += 1
                print("Num:%d" %flag)
                print(weight)
        except tf.errors.OutOfRangeError:
            print("Down.")
        finally:
            coord.request_stop()
        coord.join(threads)

        for num in range(5):
            w[num] = weight[num] / sum(weight)
        print(w)


if __name__ == '__main__':
    main()
