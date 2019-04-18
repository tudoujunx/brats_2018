import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.losses import dice_coef_loss

import tensorflow as tf
from matplotlib import pyplot as plt


CLASS_NUM = 2
BATCH_NUM = 10
EPOCHS = 5
learning_rate = 1e-5

train_save_dir = r'..\data_set\pre_data\brats_2018_keras\dataset'
cv_save_dir = r'..\data_set\pre_data\brats_2018_keras\cv'
model_save_dir = r'..\data_set\pre_data\brats_2018_keras\model'

model_name = 'unet_keras26w_{epoch:02d}-{categorical_accuracy:.2f}.hdf5'


class unet():

    def __init__(self):

        print('New network.')
        self.batch_num = None

    @staticmethod
    def conv2d_bn_relu(x, filters, kernel_size, strides=(1, 1), padding='same', activation='relu'):

        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      kernel_initializer='glorot_normal')(x)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
        return conv

    @staticmethod
    def conv2dt_bn_relu(x, filters, kernel_size, strides=(2, 2), padding='valid', activation='relu'):
        convt = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                kernel_initializer='glorot_normal')(x)
        convt = BatchNormalization()(convt)
        convt = Activation(activation)(convt)
        return convt


    def build_unet(self, input_size=(BATCH_NUM, 240, 240, 4)):

        inputs = Input(batch_shape=input_size)

        conv1 = self.conv2d_bn_relu(inputs, 64, (3, 3))
        conv1 = self.conv2d_bn_relu(conv1, 64, (3, 3))
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        drop1 = Dropout(0.5)(pool1)

        conv2 = self.conv2d_bn_relu(drop1, 128, (3, 3))
        conv2 = self.conv2d_bn_relu(conv2, 128, (3, 3))
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        drop2 = Dropout(0.5)(pool2)

        conv3 = self.conv2d_bn_relu(drop2, 256, (3, 3))
        conv3 = self.conv2d_bn_relu(conv3, 256, (3, 3))
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        drop3 = Dropout(0.5)(pool3)

        conv4 = self.conv2d_bn_relu(drop3, 512, (3, 3))
        conv4 = self.conv2d_bn_relu(conv4, 512, (3, 3))
        pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)
        drop4 = Dropout(0.5)(pool4)

        conv5 = self.conv2d_bn_relu(drop4, 1024, (3, 3))
        conv5 = self.conv2d_bn_relu(conv5, 1024, (3, 3))
        convt5 = self.conv2dt_bn_relu(conv5, 512, (2, 2))
        drop5 = Dropout(0.5)(convt5)

        merge6 = concatenate([conv4, drop5], axis=-1)
        conv6 = self.conv2d_bn_relu(merge6, 512, (3, 3))
        conv6 = self.conv2d_bn_relu(conv6, 512, (3, 3))
        convt6 = self.conv2dt_bn_relu(conv6, 256, (2, 2))
        drop6 = Dropout(0.5)(convt6)

        merge7 = concatenate([conv3, drop6], axis=-1)
        conv7 = self.conv2d_bn_relu(merge7, 256, (3, 3))
        conv7 = self.conv2d_bn_relu(conv7, 256, (3, 3))
        convt7 = self.conv2dt_bn_relu(conv7, 128, (2, 2))
        drop7 = Dropout(0.5)(convt7)

        merge8 = concatenate([conv2, drop7], axis=-1)
        conv8 = self.conv2d_bn_relu(merge8, 128, (3, 3))
        conv8 = self.conv2d_bn_relu(conv8, 128, (3, 3))
        convt8 = self.conv2dt_bn_relu(conv8, 64, (2, 2))
        drop8 = Dropout(0.5)(convt8)

        merge9 = concatenate([conv1, drop8], axis=-1)
        conv9 = self.conv2d_bn_relu(merge9, 64, (3, 3))
        conv9 = self.conv2d_bn_relu(conv9, 64, (3, 3))
        outputs = Conv2D(filters=CLASS_NUM, kernel_size=(1, 1), strides=(1, 1), padding='valid' ,activation='softmax')(conv9)

        model = Model(input=inputs, output=outputs, name='u_net')
        model.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      #loss='categorical_crossentropy',
                      loss=dice_coef_loss,
                      metrics=['categorical_accuracy'])
        model.summary()

        return model


def data_genarater(data_dir, batch_num=3, img_height=240, img_wide=240, image_channel=4, class_num=4, is_train=True):
    img_dir_list = os.listdir(data_dir + r'\image')
    label_dir_list = os.listdir(data_dir + r'\label')
    assert len(img_dir_list) == len(label_dir_list)

    index = np.arange(len(img_dir_list))
    print(len(img_dir_list))
    image = np.zeros(shape=(batch_num, img_height, img_wide, image_channel), dtype=np.uint8)
    label = np.zeros(shape=(batch_num, img_height, img_wide), dtype=np.uint8)
    i = 0
    j = 0
    epoch = 0
    while True:
        if i >= len(index):
            i = 0
        if i == 0:
            if is_train is True:
                np.random.shuffle(index)
            epoch += 1

        image[j] = np.load(os.path.join(data_dir + r'\image', img_dir_list[index[i]]))
        label[j] = np.load(os.path.join(data_dir + r'\label', label_dir_list[index[i]]))

        i += 1
        j += 1
        if j < batch_num:
            continue
        else:
            j = 0
        if class_num == 2:
            label = np.where(label>0, 1, label)
        if is_train is True:
            label_onehot = np.eye(class_num)[label.reshape(-1)]
            label_onehot = label_onehot.reshape((batch_num, img_height, img_wide, class_num))
            yield (image, label_onehot)
        else:
            yield (image, label)


def train():

    net = unet()
    model = net.build_unet()
    model_checkpoint = ModelCheckpoint(os.path.join(model_save_dir, model_name),
                                       monitor='loss', verbose=1, save_best_only=True)

    H = model.fit_generator(
        generator=data_genarater(train_save_dir, BATCH_NUM, 240, 240, 4, CLASS_NUM),
        steps_per_epoch=266705/BATCH_NUM,
        epochs=EPOCHS,
        validation_data=data_genarater(cv_save_dir, BATCH_NUM, 240, 240, 4, CLASS_NUM),
        validation_steps=3137/BATCH_NUM,
        callbacks=[model_checkpoint]
    )
    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["categorical_accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_categorical_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(r"..\data_set\pre_data\brats_2018_keras\model26w-batch5-epoch10.png")


def cross_validation(modelname):
    model = load_model(os.path.join(model_save_dir, modelname))
    for file in os.listdir(cv_save_dir + '\predict'):
        os.remove(os.path.join(cv_save_dir + '\predict', file))
    i = 0

    for (cv_image, cv_label) in data_genarater(cv_save_dir, BATCH_NUM, 240, 240, 4, CLASS_NUM, False):
        # (cv_image, cv_label) = data_genarater(cv_save_dir, 1, 240, 240, 4, 4, False)

        cv_predict = model.predict(cv_image)
        cv_predict = np.argmax(cv_predict, axis=-1)
        for j in range(BATCH_NUM):
            pre_name = str(i) + '-' + str(j)
            cv.imwrite(os.path.join(cv_save_dir + '\predict', pre_name + 'pre.png'), cv_predict[j].astype(float)*255/CLASS_NUM)
            cv.imwrite(os.path.join(cv_save_dir + '\predict', pre_name + 'ori.png'), cv_label[j].astype(float)*255/CLASS_NUM)
        i += 1
        if i >= 3137/BATCH_NUM:
            break


def read_data():
    i = 0
    for (image, label) in data_genarater(train_save_dir, 1, 240, 240, 4, 4, False):
        cv.imwrite(os.path.join(train_save_dir + '\iii', str(i) + 'img.png'),
                   image[0])
        cv.imwrite(os.path.join(train_save_dir + '\lll', str(i) + 'mask.png'),
                   label[0].astype(float) * 255 / CLASS_NUM)
        i += 1
        if(i>260000):
            break


if __name__ == '__main__':
    # net = unet()
    # net.build_unet()
    # read_data()
    # train()
    # cross_validation('unet_keras26w_01-0.95.hdf5')

