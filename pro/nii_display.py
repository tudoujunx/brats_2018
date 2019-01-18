import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

image_dir = "..\data_set\pre_data\MICCAI_BraTS_2018_Data_Training\HGG\Brats18_2013_2_1\Brats18_2013_2_1_t1ce.nii.gz"
img_width, img_height, slices = 240, 240, 155


def nothing(x):
    pass


def read_nii_data():
    nii_data = nib.load(image_dir)
    affine = nii_data.affine
    image_data = np.asarray(nii_data.dataobj)
    max, min = image_data.max(), image_data.min()
    image = ((image_data - min)/(max - min)) * 255
    return image.astype(np.uint8), affine


def image_display():
    img, _ = read_nii_data()

    cv.namedWindow('IMAGE')
    cv.createTrackbar('slice', 'IMAGE', 0, slices-1, nothing)
    while True:
        slice = cv.getTrackbarPos('slice', 'IMAGE')
        cv.imshow('IMAGE', img[:, :, slice])
        cv.waitKey(5)


    cv.destroyAllWindows()


if __name__ == '__main__':
    image_display()



