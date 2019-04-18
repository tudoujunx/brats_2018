import os
import nibabel as nib
import numpy as np
# import tensorflow as tf
# import csv
from albumentations import (Compose, OneOf, PadIfNeeded, RandomRotate90, VerticalFlip, ElasticTransform,
                            GridDistortion, OpticalDistortion)
import cv2 as cv
import SimpleITK as sitk
import gc


data_dir = r'..\data_set\pre_data\MICCAI_BraTS_2018_Data_Training'
train_save_dir = r'..\data_set\pre_data\brats_2018_keras\train'
cv_save_dir = r'..\data_set\pre_data\brats_2018_keras\cv'
dataset_dir = r'..\data_set\pre_data\brats_2018_keras\dataset'

img_width, img_height, channels, slices = 240, 240, 4, 155

"""
清空文件夹：
shutil.rmtree(train_save_dir)
os.mkdir(train_save_dir)
shutil.rmtree(cv_save_dir)
os.mkdir(cv_save_dir)
"""



def n4_bias_field_correct(inputdir, outputdir=r'..\data_set\pre_data\brats2018_n4'):
    # sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(1000000000)
    for root, dirs, files in os.walk(inputdir):
        print(root, dirs, files)  # 当前目录路径

        if len(files) == 5:
            outputpath = outputdir+'\\' + root.split("\\")[-1]
            if not os.path.isdir(outputpath):
                os.makedirs(outputpath)
            for file in files:
                if 'flair' in file:
                    inputname = os.path.join(root, file)
                    outputname = os.path.join(outputpath, 'out' + file)

                    input_image = sitk.ReadImage(inputname)
                    indata = sitk.GetArrayFromImage(input_image)
                    mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
                    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    output_image = corrector.Execute(input_image, mask_image)
                    output_image = sitk.Cast(output_image, sitk.sitkInt16)
                    sitk.WriteImage(output_image, outputname)


def load_data(data_dir):
    for root, dirs, files in os.walk(data_dir):
        print(root, dirs, files)  # 当前目录路径

        if len(files) == 5:
            for file in files:
                img = nib.load(os.path.join(root, file))
                img_data = img.get_fdata()
                img_data = img_data.astype(int)
                if '_seg' in file:
                    img_data = np.where(img_data < 3, img_data, 3)
                    label = img_data.astype('int')
                if '_flair' in file:
                    flair_img = img_data
                if '_t1.' in file:
                    t1_img = img_data
                if '_t1ce' in file:
                    t1ce_img = img_data
                if '_t2' in file:
                    (filename, extension) = os.path.splitext(file)
                    patient_name = filename.split('.')[0].strip('t2')
                    t2_img = img_data

            image = [flair_img, t1_img, t1ce_img, t2_img]
            image = np.transpose(image, (1, 2, 0, 3))
            max = image.max()
            min = image.min()
            image = 255 * (image - min) / (max - min)
            image = image.astype('uint8')

            chance = np.random.randint(100)
            if chance < 10:
                save_dir = cv_save_dir
            else:
                save_dir = train_save_dir

            for i in range(slices):
                if image[:, :, :, i].max() == 0 and label[:, :, i].max() == 0:
                    continue
                np.save(os.path.join(save_dir + '\image', patient_name + str(i) + '.npy'), image[:, :, :, i])
                np.save(os.path.join(save_dir + '\label', patient_name + str(i) + '.npy'), label[:, :, i])

        print('----------------------------------')
    print('Load success')


def crop_image(data_dirs, save_dir):
    image_dir = data_dirs + '\image'
    label_dir = data_dirs + '\label'
    i = 0
    for file in os.listdir(image_dir):
        ori_image = np.load(os.path.join(image_dir, file))
        ori_label = np.load(os.path.join(label_dir, file))
        coords = np.where(ori_image.sum(-1) > 0)
        if len(coords[0]) > 1:
            bound = np.array((np.min(coords[0]), np.max(coords[0])+1, np.min(coords[1]), np.max(coords[1])+1))

        assert bound[1] > bound[0] and bound[3] > bound[2]
        if bound[1]-bound[0] < 10 or bound[3]-bound[2] < 10:
            continue
        crop_img = ori_image[bound[0]:bound[1], bound[2]:bound[3]]
        crop_label = ori_label[bound[0]:bound[1], bound[2]:bound[3]]

        np.save(os.path.join(save_dir + '\image_crop', str(i) + '.npy'), crop_img)
        np.save(os.path.join(save_dir + '\label_crop', str(i) + '.npy'), crop_label)
        i += 1


def produce_dataset(data_dirs, save_dir, times=1):
    image_dir = data_dirs + '\image_crop'
    label_dir = data_dirs + '\label_crop'

    aug = Compose([PadIfNeeded(min_height=img_width, min_width=img_height, p=1),
                  VerticalFlip(p=0.5), RandomRotate90(p=0.5),
                   OneOf([
                       ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                       GridDistortion(p=0.5),
                       OpticalDistortion(p=0.5, distort_limit=1, shift_limit=0.5)
                   ], p=0.8)])
    i = 0
    for epoch in range(times):
        for file in os.listdir(image_dir):
            ori_image = np.load(os.path.join(image_dir, file))
            ori_label = np.load(os.path.join(label_dir, file))

            assert ori_image.dtype == 'uint8'
            assert ori_label.dtype == 'int'

            augmented = aug(image=ori_image, mask=ori_label)

            image = augmented['image']
            label = augmented['mask']

            if label.max() == 0:
                a = np.random.rand()
                if a < 0.5:
                    continue

            np.save(os.path.join(save_dir + '\image', str(i) + '.npy'), image)
            np.save(os.path.join(save_dir + '\label', str(i) + '.npy'), label)

            i += 1
        print('epoch = ' + str(epoch))
    print(' Load over!\n i = ' + str(i))


def test(data_dirs):
    image_dir = data_dirs + '\image'
    label_dir = data_dirs + '\label'
    i = 0
    for file in os.listdir(image_dir):
        ori_image = np.load(os.path.join(image_dir, file))
        ori_label = np.load(os.path.join(label_dir, file))
        if ori_image.max() == 0:
            print(file)
        if ori_label.max() == 0:
            i += 1
            if i%100 == 0:
                print(i)
    print('Over! ' + str(i))






if __name__ == '__main__':
    # crop_image(train_save_dir, dataset_dir)
    # produce_dataset(dataset_dir, dataset_dir, 10)
    n4_bias_field_correct(data_dir)
    test(dataset_dir)








