import tensorflow as tf
# import tensorflow.contrib.rnn as rnn
import argparse
import os
import numpy as np
from pro import data_load_and_parsing

# DIR = 'D:/program/python/test/test/image/Brats18_2013_2_1'
train_load_dir = 'D:/program/pro_code/brats_2018/data_set/pre_data/output_train.tfrecords'
train_load_dir = 'D:/program/python/test/BratsData/predata/output_train.tfrecords'

TRAIN_SET_NAME = 'output_train.tfrecords'
VALIDATION_SET_NAME = 'output_cv.tfrecords'
TEST_SET_NAME = 'output_test.tfrecords'
ORIGIN_PREDICT_DIRECTORY = '../data_set/test'
PREDICT_SAVED_DIRECTORY = '../data_set/predictions'

INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL, slices = 240, 240, 4, 155
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 240, 240, 1
EPOCH_NUM = 15
TRAIN_BATCH_SIZE = 3
VALIDATION_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1
EPS = 1e-6
FLAGS = None
CLASS_NUM = 4
CHECK_POINT_PATH = None
learning_rate = 1e-5
min_after_dequeue = 15


class Net:

    def __init__(self):
        print('New Network.')
        self.input_image = None
        self.input_label = None
        self.cast_image = None
        self.cast_label = None
        self.keep_prob = None
        self.lamb = None
        self.result_expand = None
        self.loss, self.balance_loss, self.loss_mean, self.loss_all, self.train_step = [None] * 5
        self.loss_weight = None
        self.prediction, self.correct_prediction, self.accuracy, self.final_prediction = [None] * 4
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contract_layer = {}
        self.w = {}
        self.b = {}
        self.is_traing = None
        self.lr = None
        self.iou = None

    def init_w(self, shape, name):
        with tf.name_scope('init_w'):
            stddev = 1.0
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32),
                            name=name)
            tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
            return w

    @staticmethod
    def copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling):
        result_from_contract_layer_crop = result_from_contract_layer
        return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=-1)

    @staticmethod
    def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
        from tensorflow.python.training.moving_averages import assign_moving_average

        with tf.variable_scope(name):
            params_shape = x.shape[-1:]
            moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                         trainable=False)

            def mean_var_with_update():
                mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)),
                                                                     name='moments')
                with tf.control_dependencies([
                    assign_moving_average(moving_mean, mean_this_batch, decay),
                    assign_moving_average(moving_var, variance_this_batch, decay)
                ]):
                    return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
            if affine:  # 如果要用beta和gamma进行放缩
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                                   variance_epsilon=eps)
            else:
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                                   variance_epsilon=eps)
            return normed

    def set_up_network(self, batch_size):

        # input
        with tf.name_scope('input'):
            self.input_image = tf.placeholder(
                dtype=tf.float32,
                shape=[batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL],
                name='input_image'
            )
            self.input_label = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE],
                name='input_label'
            )
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
            self.is_traing = tf.placeholder(dtype=tf.bool, name='is_training')
            normed_batch = self.batch_norm(x=self.input_image, is_training=self.is_traing, name='input')

        # layer 1
        with tf.name_scope('layer_1'):
            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, INPUT_IMG_CHANNEL, 64], name='w_1')
            # self.b[1] = self.init_b(shape=[64], name='b_1')
            result_conv_1 = tf.nn.conv2d(
                input=normed_batch, filter=self.w[1],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_1_conv_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias'), name='relu_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[2] = self.init_w(shape=[3, 3, 64, 64], name='w_2')
            # self.b[2] = self.init_b(shape=[64], name='b_2')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[2],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_1_conv_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[2], name='add_bias'), name='relu_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[1] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

            # layer 2
        with tf.name_scope('layer_2'):
            # conv_1
            self.w[3] = self.init_w(shape=[3, 3, 64, 128], name='w_3')
            # self.b[3] = self.init_b(shape=[128], name='b_3')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[3],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_2_conv_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[3], name='add_bias'), name='relu_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[4] = self.init_w(shape=[3, 3, 128, 128], name='w_4')
            # self.b[4] = self.init_b(shape=[128], name='b_4')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[4],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_2_conv_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[4], name='add_bias'), name='relu_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[2] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

            # layer 3
        with tf.name_scope('layer_3'):
            # conv_1
            self.w[5] = self.init_w(shape=[3, 3, 128, 256], name='w_5')
            # self.b[5] = self.init_b(shape=[256], name='b_5')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[5],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_3_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[5], name='add_bias'), name='relu_1')

            # conv_2
            self.w[6] = self.init_w(shape=[3, 3, 256, 256], name='w_6')
            # self.b[6] = self.init_b(shape=[256], name='b_6')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[6],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_3_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[6], name='add_bias'), name='relu_2')
            self.result_from_contract_layer[3] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

            # layer 4
        with tf.name_scope('layer_4'):
            # conv_1
            self.w[7] = self.init_w(shape=[3, 3, 256, 512], name='w_7')
            # self.b[7] = self.init_b(shape=[512], name='b_7')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[7],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_4_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[7], name='add_bias'), name='relu_1')

            # conv_2
            self.w[8] = self.init_w(shape=[3, 3, 512, 512], name='w_8')
            # self.b[8] = self.init_b(shape=[512], name='b_8')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[8],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_4_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[8], name='add_bias'), name='relu_2')
            self.result_from_contract_layer[4] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

            # layer 5 (bottom)
        with tf.name_scope('layer_5'):
            # conv_1
            self.w[9] = self.init_w(shape=[3, 3, 512, 1024], name='w_9')
            # self.b[9] = self.init_b(shape=[1024], name='b_9')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[9],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_5_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[9], name='add_bias'), name='relu_1')

            # conv_2
            self.w[10] = self.init_w(shape=[3, 3, 1024, 1024], name='w_10')
            # self.b[10] = self.init_b(shape=[1024], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[10],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_5_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[10], name='add_bias'), name='relu_2')

            # up sample
            self.w[11] = self.init_w(shape=[2, 2, 512, 1024], name='w_11')
            # self.b[11] = self.init_b(shape=[512], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[11],
                output_shape=[batch_size, 30, 30, 512],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_5_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')
            # result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[11], name='add_bias'), name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

            # layer 6
        with tf.name_scope('layer_6'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[4], result_from_upsampling=result_dropout)
            # print(result_merge)

            # conv_1
            self.w[12] = self.init_w(shape=[3, 3, 1024, 512], name='w_12')
            # self.b[12] = self.init_b(shape=[512], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[12],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_6_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[12], name='add_bias'), name='relu_1')

            # conv_2
            self.w[13] = self.init_w(shape=[3, 3, 512, 512], name='w_13')
            # self.b[13] = self.init_b(shape=[512], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[13],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_6_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[13], name='add_bias'), name='relu_2')
            # print(result_relu_2.shape[1])

            # up sample
            self.w[14] = self.init_w(shape=[2, 2, 256, 512], name='w_14')
            # self.b[14] = self.init_b(shape=[256], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[14],
                output_shape=[batch_size, 60, 60, 256],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_6_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')
            # result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[14], name='add_bias'), name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

            # layer 7
        with tf.name_scope('layer_7'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_dropout)

            # conv_1
            self.w[15] = self.init_w(shape=[3, 3, 512, 256], name='w_15')
            # self.b[15] = self.init_b(shape=[256], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[15],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_7_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[15], name='add_bias'), name='relu_1')

            # conv_2
            self.w[16] = self.init_w(shape=[3, 3, 256, 256], name='w_16')
            # self.b[16] = self.init_b(shape=[256], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[16],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_7_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[16], name='add_bias'), name='relu_2')

            # up sample
            self.w[17] = self.init_w(shape=[2, 2, 128, 256], name='w_17')
            # self.b[17] = self.init_b(shape=[128], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[17],
                output_shape=[batch_size, 120, 120, 128],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_7_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')
            # result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[17], name='add_bias'), name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

            # layer 8
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[2], result_from_upsampling=result_dropout)

            # conv_1
            self.w[18] = self.init_w(shape=[3, 3, 256, 128], name='w_18')
            # self.b[18] = self.init_b(shape=[128], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[18],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_8_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[18], name='add_bias'), name='relu_1')

            # conv_2
            self.w[19] = self.init_w(shape=[3, 3, 128, 128], name='w_19')
            # self.b[19] = self.init_b(shape=[128], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[19],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_8_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[19], name='add_bias'), name='relu_2')

            # up sample
            self.w[20] = self.init_w(shape=[2, 2, 64, 128], name='w_20')
            # self.b[20] = self.init_b(shape=[64], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[20],
                output_shape=[batch_size, 240, 240, 64],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_8_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')
            # result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[20], name='add_bias'), name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

            # layer 9
        with tf.name_scope('layer_9'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[1], result_from_upsampling=result_dropout)

            # conv_1
            self.w[21] = self.init_w(shape=[3, 3, 128, 64], name='w_21')
            # self.b[21] = self.init_b(shape=[64], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[21],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_9_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')
            # result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[21], name='add_bias'), name='relu_1')

            # conv_2
            self.w[22] = self.init_w(shape=[3, 3, 64, 64], name='w_22')
            # self.b[22] = self.init_b(shape=[64], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[22],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_9_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[22], name='add_bias'), name='relu_2')

            # convolution to [batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
            self.w[23] = self.init_w(shape=[1, 1, 64, CLASS_NUM], name='w_11')
            # self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
            result_conv_3 = tf.nn.conv2d(
                input=result_relu_2, filter=self.w[23],
                strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_3, is_training=self.is_traing, name='layer_9_conv_3')

            self.prediction = normed_batch
            # self.prediction = tf.nn.relu(tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='relu_3')
            # self.prediction = result_conv_3
            # self.prediction = tf.nn.sigmoid(x=tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='sigmoid_1')
            # self.prediction = tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias')
            self.final_prediction = tf.argmax(input=self.prediction, axis=3, output_type=tf.int32)
        # print(self.prediction)
        # print(self.final_prediction)
        # print(self.input_label)

        # softmax loss
        with tf.name_scope('softmax_loss'):
            # using one-hot labels
            # self.loss = \
            #     tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')

            # not using one-hot

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label,
                                                                       logits=self.prediction,
                                                                       name='loss')
            """
            self.loss_weight = tf.reshape(
                tf.Variable(initial_value=[0.001, 1, 1, 1], dtype=tf.float32, name='lossweight'), [1, 1, 1, 4])
            pre_softmax = tf.nn.softmax(self.prediction, axis=-1)
            label_vector = tf.reshape(self.input_label, [batch_size*INPUT_IMG_HEIGHT*INPUT_IMG_WIDE],
                                      name='labelvector')
            with tf.device('/cpu:0'):
                label_vector_onehot = tf.one_hot(label_vector, depth=CLASS_NUM, axis=-1)

            label_onehot = tf.reshape(label_vector_onehot, [batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, CLASS_NUM],
                                      name='labelonehot')

            self.loss = tf.reduce_sum(tf.multiply((-tf.log(pre_softmax)), label_onehot), axis=-1, name='loss')
            """
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        # accuracy
        with tf.name_scope('accuracy'):
            # using one-hot
            # self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.cast_label, axis=3))

            # not using one-hot
            self.correct_prediction = \
                tf.equal(self.final_prediction, self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)
            self.iou = tf.metrics.mean_iou(labels=self.input_label, predictions=self.final_prediction,
                                           num_classes=4, name='mean_iou')

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            #  global_steps = tf.Variable(0, trainable=False)
            # self.lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_steps, decay_steps=1,
            #                                      decay_rate=0.9, staircase=True, name='learning_rate')

            self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_all)

    def train(self):
        train_file_path = os.path.join(FLAGS.data_dir, TRAIN_SET_NAME)
        train_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM, shuffle=True
        )
        ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        train_images, train_labels = data_load_and_parsing.parsing_data(train_image_filename_queue, TRAIN_BATCH_SIZE)
        tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.image("prediction", self.prediction)
        tf.summary.scalar('accuracy', self.accuracy)
        # tf.summary.scalar('learning_rate', self.LR)
        merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver(max_to_keep=1)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # plt.figure(1)

            try:
                epoch = 1
                h = 1
                while not coord.should_stop():
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    example_batch, label_batch = sess.run([train_images, train_labels])    # 在会话中取出image和label
                    # plt.imshow(label_batch[0, :, :, 50], origin="lower")
                    # plt.show()
                    print('training_step %d' % epoch)
                    epoch += 1
                    for i in range(slices):
                        example = example_batch[:, :, :, :, i]
                        label = label_batch[:, :, :, i]
                        # label = tf.one_hot(label, 4)
                        # print(label)
                        flag = 1
                        # label_weight = np.zeros(CLASS_NUM)
                        for pix in label.flat:
                            # label_weight[pix] += 1
                            if pix != 0:
                                flag = 0
                                break
                        if flag:
                            continue
                        # label_weight = label_weight[::-1]/(EPOCH_NUM*INPUT_IMG_WIDE*INPUT_IMG_HEIGHT)
                        # plt.imshow(label[0, :, :])
                        # plt.show()
                        """
                        pre, pre_data, loss = sess.run(
                            [tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.prediction, self.loss],
                            feed_dict={
                                self.input_image: example, self.input_label: label, self.keep_prob: 0.5,
                                self.lamb: 0.01, self.is_traing: True}
                        )
                        # plt.imshow(pre[0, :, :])
                        # plt.show()
                        """
                        """
                        loss_weight = tf.Variable(initial_value=[0.002, 1, 1, 1], dtype=tf.float32,
                                                       name='lossweight')
                        pre = sess.run(self.prediction, feed_dict={self.input_image: example,
                                                         self.input_label: label,
                                                         self.keep_prob: 0.5,
                                                         self.lamb: 0.01,
                                                         self.is_traing: True})

                        pre_softmax = tf.nn.softmax(pre, axis=-1)
                        label_vector = tf.reshape(label, [EPOCH_NUM * INPUT_IMG_HEIGHT * INPUT_IMG_WIDE],
                                                  name='labelvector')
                        label_vector_onehot = tf.one_hot(label_vector, depth=CLASS_NUM, axis=-1)
                        label_onehot = tf.reshape(label_vector_onehot,
                                                  [EPOCH_NUM, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, CLASS_NUM],
                                                  name='labelonehot')
                        la = sess.run(label_onehot)
                        los = tf.reduce_sum(tf.multiply((-tf.log(pre_softmax)), label_onehot), axis=-1)

                        ls = sess.run(los)
                        los = sess.run(self.loss,
                                              feed_dict={self.input_image: example,
                                                         self.input_label: label,
                                                         self.keep_prob: 0.5,
                                                         self.lamb: 0.01,
                                                         self.is_traing: True})
"""
                        lo, acc, summary_str, loss = sess.run(
                            [self.loss_mean, self.accuracy, merged_summary, self.loss],
                            feed_dict={
                                self.input_image: example, self.input_label: label, self.keep_prob: 0.5,
                                self.lamb: 0.5, self.is_traing: True}
                        )
                        summary_writer.add_summary(summary_str, epoch)
                        # print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))

                        sess.run(
                            [self.train_step],
                            feed_dict={
                                self.input_image: example, self.input_label: label, self.keep_prob: 0.5,
                                self.lamb: 0.5, self.is_traing: True}
                        )
                        h += 1
                        if h % 10 == 0:
                            print('num %d, loss: %.6f and accuracy: %.6f' % (h, lo, acc))
                    if epoch % 10 == 0:
                        all_parameters_saver.save(sess=sess, save_path=ckpt_path)
                        print('Save step %d' % epoch)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                all_parameters_saver.save(sess=sess, save_path=ckpt_path)
                print('Done saving final step %d' % epoch)
                coord.request_stop()
            coord.join(threads)
        print("Done training")

    def validate(self):
        # import cv2
        # import numpy as np
        # ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        # # mydata = DataProcess(INPUT_IMG_HEIGHT, INPUT_IMG_WIDE)
        # # imgs_train, imgs_mask_train = mydata.load_my_train_data()
        # all_parameters_saver = tf.train.Saver()
        # my_set_image = cv2.imread('../data_set/train.tif', flags=0)
        # my_set_label = cv2.imread('../data_set/label.tif', flags=0)
        # my_set_image.astype('float32')
        # my_set_label[my_set_label <= 128] = 0
        # my_set_label[my_set_label > 128] = 1
        # with tf.Session() as sess:
        # 	sess.run(tf.global_variables_initializer())
        # 	sess.run(tf.local_variables_initializer())
        # 	all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
        # 	image, acc = sess.run(
        # 		fetches=[self.prediction, self.accuracy],
        # 		feed_dict={
        # 				self.input_image: my_set_image, self.input_label: my_set_label,
        # 				self.keep_prob: 1.0, self.lamb: 0.004}
        # 	)
        # image = np.argmax(a=image[0], axis=2).astype('uint8') * 255
        # # cv2.imshow('predict', image)
        # # cv2.imshow('o', np.asarray(a=image[0], dtype=np.uint8) * 100)
        # # cv2.waitKey(0)
        # cv2.imwrite(filename=os.path.join(FLAGS.model_dir, 'predict.jpg'), img=image)
        # print(acc)
        # print("Done test, predict image has been saved to %s" % (os.path.join(FLAGS.model_dir, 'predict.jpg')))
        validation_file_path = os.path.join(FLAGS.data_dir, VALIDATION_SET_NAME)
        validation_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(validation_file_path), num_epochs=1, shuffle=True)
        ckpt_path = os.path.join(FLAGS.model_dir + "/model.ckpt")
        validation_images, validation_labels = data_load_and_parsing.parsing_data(validation_image_filename_queue, VALIDATION_BATCH_SIZE)
        # tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                h = 1
                while not coord.should_stop():
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    example_batch, label_batch = sess.run([validation_images, validation_labels])  # 在会话中取出image和label
                    # print(label)
                    print('img %d' % epoch)
                    for i in range(slices):
                        example = example_batch[:, :, :, :, i]
                        label = label_batch[:, :, :, i]
                        acc, pre, iou = sess.run(
                            [self.accuracy, self.final_prediction, self.iou],
                            feed_dict={
                                self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                                self.lamb: 0.0, self.is_traing: False}
                        )
                        h += 1
                    # summary_writer.add_summary(summary_str, epoch)
                    # print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                        if h % 1 == 0:
                            print('num %d, accuracy: %.6f' % (h, acc))
                            print(iou)
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('Done validating -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print('Done validating')

    def test(self):
        """
        import nibabel as nib
        # data_load_and_parsing.test_data_load()
        test_file_path = os.path.join(FLAGS.data_dir, TEST_SET_NAME)
        test_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(test_file_path), num_epochs=1, shuffle=False)
        ckpt_path = os.path.join(FLAGS.model_dir + "/model.ckpt")
        test_images, test_name = data_load_and_parsing.parsing_test_data(test_image_filename_queue, TEST_BATCH_SIZE)
        # tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            pre_image = np.zeros(shape=(INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, slices))
            try:
                epoch = 0
                while not coord.should_stop():
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    image_batch, name = sess.run([test_images, test_name])  # 在会话中取出image和name
                    name = name.decode('utf-8')
                    # print(label)

                    affine = np.diag([1, 1, 1, 1])
                    for i in range(slices):

                        pre_image[:, :, i] = sess.run(
                            self.final_prediction[0, :, :],
                            feed_dict={
                                self.input_image: image_batch[:, :, :, :, i],
                                self.keep_prob: 1.0, self.lamb:  0.0,
                                self.is_traing: False
                            }
                        )
                    epoch += 1
                    img_data = np.where(pre_image == 3, 4, pre_image)
                    nii_image = nib.Nifti1Image(img_data, affine)
                    nib.save(nii_image, os.path.join(FLAGS.data_dir + '/prediction/' + name + '.nii.gz'))

                    if epoch % 1 == 0:
                        print('num %d ' % (epoch))
            except tf.errors.OutOfRangeError:
                print('Done testing -- epoch limit reached \n ')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print('Done testing')
        """
        import nibabel as nib
        ckpt_path = os.path.join(FLAGS.model_dir + "/model.ckpt")
        all_parameters_saver = tf.train.Saver()
        test_data_dir = '../data_set/pre_data/MICCAI_BraTS_2018_Data_Validation'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)

            pre_image = np.zeros(shape=(INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, slices))
            flair_img, t1_img, t1ce_img, t2_img = [np.zeros(shape=(INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, slices))] * 4
            affine = np.diag([1, 1, 1, 1])
            epoch = 1
            for root, dirs, files in os.walk(test_data_dir):
                print(root, dirs, files)  # 当前目录路径

                if len(files):
                    if len(files) == 4:
                        for file in files:
                            print(file)
                            (filename, extension) = os.path.splitext(file)
                            img = nib.load(root + '/' + file)
                            # img_data = img.get_fdata()
                            img_data = np.asarray(img.dataobj)
                            affine = img.affine
                            if '_flair' in filename:
                                flair_img = img_data

                            if '_t1' in filename:
                                t1_img = img_data

                            if '_t1ce' in filename:
                                t1ce_img = img_data

                            if '_t2' in filename:
                                t2_img = img_data
                                name = filename.split('.')[0].strip('_t2')

                        image = [flair_img, t1_img, t1ce_img, t2_img]
                        image = np.transpose(image, [1, 2, 0, 3])
                        image = np.reshape(image, [1, 240, 240, 4, 155])

                        for i in range(slices):
                            image_slice = image[:, :, :, :, i]
                            pre_image[:, :, i] = sess.run(
                                self.final_prediction,
                                feed_dict={
                                    self.input_image: image_slice,
                                    self.keep_prob: 1.0, self.lamb: 0.0,
                                    self.is_traing: False
                                }
                            )

                        epoch += 1
                        img_data = np.where(pre_image == 3, 4, pre_image)
                        nii_image = nib.Nifti1Image(img_data, affine)
                        nib.save(nii_image, os.path.join(FLAGS.data_dir + '/prediction/' + name + '.nii.gz'))

                        if epoch % 1 == 0:
                            print('num %d ' % epoch + name)
        print('Done testing')


"""
    def predict(self):
        import cv2
        import glob
        import numpy as np
        # TODO 不应该这样写，应该直接读图片预测，而不是从tfrecord读取，因为顺序变了，无法对应
        predict_file_path = glob.glob(os.path.join(ORIGIN_PREDICT_DIRECTORY, '*.tif'))
        print(len(predict_file_path))
        if not os.path.lexists(PREDICT_SAVED_DIRECTORY):
            os.mkdir(PREDICT_SAVED_DIRECTORY)
        ckpt_path = CHECK_POINT_PATH
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            for index, image_path in enumerate(predict_file_path):
                # image = cv2.imread(image_path, flags=0)
                image = np.reshape(a=cv2.imread(image_path, flags=0), newshape=(1, INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
                predict_image = sess.run(
                    tf.argmax(input=self.prediction, axis=3),
                    feed_dict={
                        self.input_image: image,
                        self.keep_prob: 1.0, self.lamb: 0.004
                    }
                )
                cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.jpg' % index), predict_image[0] * 255)
        print('Done prediction')
"""


def main():
    net = Net()
    CHECK_POINT_PATH = os.path.join(FLAGS.model_dir + "/model.ckpt")
    # print(CHECK_POINT_PATH)
    # net.set_up_network(TRAIN_BATCH_SIZE)
    # net.train()
    # net.set_up_network(VALIDATION_BATCH_SIZE)
    # net.validate()
    net.set_up_network(TEST_BATCH_SIZE)
    net.test()
    # net.set_up_network(PREDICT_BATCH_SIZE)
    # net.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据地址
    parser.add_argument(
        '--data_dir', type=str, default='../data_set/pre_data',
        help='Directory for storing input data')

    # 模型保存地址
    parser.add_argument(
        '--model_dir', type=str, default='../data_set/saved_models',
        help='output model path')

    # 日志保存地址
    parser.add_argument(
        '--tb_dir', type=str, default='../data_set/logs',
        help='TensorBoard log path')

    FLAGS, _ = parser.parse_known_args()
    # write_img_to_tfrecords()
    # read_check_tfrecords()
    main()


