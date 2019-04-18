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

INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL, INPUT_IMG_DEPTH = 240, 240, 4, 155
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 240, 240, 1
EPOCH_NUM = 3
TRAIN_BATCH_SIZE = 1
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
        self.is_training = None
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
                shape=[batch_size, INPUT_IMG_DEPTH, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL],
                name='input_image'
            )

            self.input_label = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size, INPUT_IMG_DEPTH, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE],
                name='input_label'
            )

            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            normed_batch = self.batch_norm(x=self.input_image, is_training=self.is_training, name='input')

        # layer 1
        with tf.name_scope('layer_1'):

            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, 3, INPUT_IMG_CHANNEL, 64], name='w_1')
            result_conv_1 = tf.nn.conv3d(
                input=normed_batch,
                filter=self.w[1],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_1'
            )
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_1_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[2] = self.init_w(shape=[3, 3, 3, 64, 64], name='w_2')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[2],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_2'
            )
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_1_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[1] = result_relu_2

            # maxpool
            result_maxpool = tf.nn.max_pool3d(
                input=result_relu_2,
                ksize=[1, 2, 2, 2, 1],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='maxpool'
            )
            # drop out
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        # layer 2
        with tf.name_scope('layer_2'):
            # conv_1
            self.w[3] = self.init_w(shape=[3, 3, 3, 64, 128], name='w_3')
            result_conv_1 = tf.nn.conv3d(
                input=result_dropout,
                filter=self.w[3],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_1'
            )
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_2_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[4] = self.init_w(shape=[3, 3, 3, 128, 128], name='w_4')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[4],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_2'
            )
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_2_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[2] = result_relu_2

            # maxpool
            result_maxpool = tf.nn.max_pool3d(
                input=result_relu_2,
                ksize=[1, 2, 2, 2, 1],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='maxpool'
            )
            # drop out
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)


        # layer 3
        with tf.name_scope('layer_3'):
            # conv_1
            self.w[5] = self.init_w(shape=[3, 3, 3, 128, 256], name='w_5')
            result_conv_1 = tf.nn.conv3d(
                input=result_dropout,
                filter=self.w[5],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_1'
            )
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_3_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[6] = self.init_w(shape=[3, 3, 3, 256, 256], name='w_6')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[6],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_2'
            )
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_3_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[3] = result_relu_2

            # maxpool
            result_maxpool = tf.nn.max_pool3d(
                input=result_relu_2,
                ksize=[1, 2, 2, 2, 1],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='maxpool'
            )
            # drop out
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        # layer 4
        with tf.name_scope('layer_4'):
            # conv_1
            self.w[7] = self.init_w(shape=[3, 3, 3, 256, 512], name='w_7')
            result_conv_1 = tf.nn.conv3d(
                input=result_dropout,
                filter=self.w[7],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_1'
            )
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_4_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[8] = self.init_w(shape=[3, 3, 3, 512, 512], name='w_8')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[8],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='conv_2'
            )
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_4_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[4] = result_relu_2

            # maxpool
            result_maxpool = tf.nn.max_pool3d(
                input=result_relu_2,
                ksize=[1, 2, 2, 2, 1],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='maxpool'
            )
            # drop out
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        # layer 5 (bottom)
        with tf.name_scope('layer_5'):
            # conv_1
            self.w[9] = self.init_w(shape=[3, 3, 3, 512, 1024], name='w_9')
            result_conv_1 = tf.nn.conv3d(
                input=result_dropout,
                filter=self.w[9],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_1'
            )
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_5_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[10] = self.init_w(shape=[3, 3, 3, 1024, 1024], name='w_10')
            # self.b[10] = self.init_b(shape=[1024], name='b_10')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[10],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_5_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[10], name='add_bias'), name='relu_2')

            # up sample
            self.w[11] = self.init_w(shape=[2, 2, 2, 512, 1024], name='w_11')
            # self.b[11] = self.init_b(shape=[512], name='b_11')
            result_up = tf.nn.conv3d_transpose(
                value=result_relu_2,
                filter=self.w[11],
                output_shape=[batch_size, 19, 30, 30, 512],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_5_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 6
        with tf.name_scope('layer_6'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[4], result_from_upsampling=result_dropout)

            # conv_1
            self.w[12] = self.init_w(shape=[3, 3, 3, 1024, 512], name='w_12')
            result_conv_1 = tf.nn.conv3d(
                input=result_merge,
                filter=self.w[12],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_6_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[13] = self.init_w(shape=[3, 3, 3, 512, 512], name='w_13')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[13],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_6_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # up sample
            self.w[14] = self.init_w(shape=[2, 2, 2, 256, 512], name='w_14')
            result_up = tf.nn.conv3d_transpose(
                value=result_relu_2, filter=self.w[14],
                output_shape=[batch_size, 38, 60, 60, 256],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_6_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 7
        with tf.name_scope('layer_7'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_dropout)

            # conv_1
            self.w[15] = self.init_w(shape=[3, 3, 3, 512, 256], name='w_15')
            result_conv_1 = tf.nn.conv3d(
                input=result_merge,
                filter=self.w[15],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_7_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[16] = self.init_w(shape=[3, 3, 3, 256, 256], name='w_16')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[16],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_7_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # up sample
            self.w[17] = self.init_w(shape=[2, 2, 2, 128, 256], name='w_17')
            result_up = tf.nn.conv3d_transpose(
                value=result_relu_2,
                filter=self.w[17],
                output_shape=[batch_size, 77, 120, 120, 128],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_7_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 8
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[2], result_from_upsampling=result_dropout)

            # conv_1
            self.w[18] = self.init_w(shape=[3, 3, 3, 256, 128], name='w_18')
            result_conv_1 = tf.nn.conv3d(
                input=result_merge,
                filter=self.w[18],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_8_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[19] = self.init_w(shape=[3, 3, 3, 128, 128], name='w_19')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[19],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_8_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # up sample
            self.w[20] = self.init_w(shape=[2, 2, 2, 64, 128], name='w_20')
            result_up = tf.nn.conv3d_transpose(
                value=result_relu_2,
                filter=self.w[20],
                output_shape=[batch_size, 155, 240, 240, 64],
                strides=[1, 2, 2, 2, 1],
                padding='VALID', name='Up_Sample')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_8_conv_up')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 9
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[1], result_from_upsampling=result_dropout)

            # conv_1
            self.w[21] = self.init_w(shape=[3, 3, 3, 128, 64], name='w_21')
            result_conv_1 = tf.nn.conv3d(
                input=result_merge,
                filter=self.w[21],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_training, name='layer_9_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[22] = self.init_w(shape=[3, 3, 3, 64, 64], name='w_22')
            result_conv_2 = tf.nn.conv3d(
                input=result_relu_1,
                filter=self.w[22],
                strides=[1, 1, 1, 1, 1],
                padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_training, name='layer_9_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # conv_3
            self.w[23] = self.init_w(shape=[1, 1, 1, 64, CLASS_NUM], name='w_23')
            result_conv_3 = tf.nn.conv3d(
                input=result_relu_2,
                filter=self.w[23],
                strides=[1, 1, 1, 1, 1],
                padding='VALID', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_3, is_training=self.is_training, name='layer_9_conv_3')

            self.prediction = normed_batch
            self.final_prediction = tf.argmax(input=self.prediction, axis=4, output_type=tf.int32)

        with tf.name_scope('softmax_loss'):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label,
                                                                       logits=self.prediction,
                                                                       name='loss')
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        with tf.name_scope('accuracy'):
            self.correct_prediction = \
                tf.equal(self.final_prediction, self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)
            self.iou = tf.metrics.mean_iou(labels=self.input_label, predictions=self.final_prediction,
                                           num_classes=4, name='mean_iou')

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
        all_parameters_saver = tf.train.Saver()
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
                while not coord.should_stop():
                    # Run training steps or whatever
                    example_batch, label_batch = sess.run([train_images, train_labels])    # 在会话中取出image和label
                    example_batch = np.transpose(example_batch, [0, 4, 1, 2, 3])
                    label_batch = np.transpose(label_batch, [0, 3, 1, 2])
                    print('training_step %d' % epoch)
                    epoch += 1

                    lo, acc, summary_str, loss = sess.run(
                        [self.loss_mean, self.accuracy, merged_summary, self.loss],
                        feed_dict={
                            self.input_image: example_batch,
                            self.input_label: label_batch,
                            self.keep_prob: 0.5, self.lamb: 0.5, self.is_training: True}
                    )
                    summary_writer.add_summary(summary_str, epoch)
                    # print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))

                    sess.run(
                        [self.train_step],
                        feed_dict={
                            self.input_image: example_batch, self.input_label: label_batch, self.keep_prob: 0.5,
                            self.lamb: 0.5, self.is_training: True}
                    )
                    if epoch % 1 == 0:
                        print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
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


def main():
    net = Net()
    CHECK_POINT_PATH = os.path.join(FLAGS.model_dir + "/model.ckpt")
    # print(CHECK_POINT_PATH)
    net.set_up_network(TRAIN_BATCH_SIZE)
    net.train()
    # net.set_up_network(VALIDATION_BATCH_SIZE)
    # net.validate()
    # net.set_up_network(TEST_BATCH_SIZE)
    # net.test()
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
