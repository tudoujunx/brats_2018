import SimpleITK as sitk
import cv2 as cv
import numpy as np
import matplotlib as plt
import threading

dir1 = r"D:\program\pro_code\brats_2018\data_set\pre_data\brats_2015\training\HGG\brats_2013_pat0001_1\VSD.Brain.XX.O.MR_Flair.54512.mha"
dir2 = r"D:\program\pro_code\brats_2018\data_set\pre_data\brats_2015\training\HGG\brats_2013_pat0001_1\VSD.Brain_3more.XX.O.OT.54517.mha"


def read_img(dir):
    img = sitk.ReadImage(dir)
    img = sitk.GetArrayFromImage(img)
    img = np.transpose(img, [1, 2, 0])
    max, min = img.max(), img.min()
    img = ((img - min) / (max - min)) * 255
    return img.astype(np.uint8)


def nothing(x):
    pass


def show_img(img):
    cv.namedWindow('IMAGE')
    cv.createTrackbar('slice', 'IMAGE', 0, 154, nothing)
    slice = 0
    while slice != 154:
        slice = cv.getTrackbarPos('slice', 'IMAGE')
        cv.imshow('IMAGE', img[:, :, slice])
        cv.waitKey(5)
    cv.destroyAllWindows()


def show_slice_img(img):
    pass


if __name__ == '__main__':
    img1 = read_img(dir1)
    show_img(img1)
    img2 = read_img(dir2)
    show_img(img2)



