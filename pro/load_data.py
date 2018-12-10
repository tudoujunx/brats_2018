import os
import nibabel as nib
import numpy as np
import tensorflow as tf
# import csv

DIR = 'D:/program/pro_code/brats_2018/data_set/pre_data/MICCAI_BraTS_2018_Data_Training'
train_load_dir = 'D:/program/pro_code/brats_2018/data_set/pre_data/output_train.tfrecords'
cv_load_dir = 'D:/program/pro_code/brats_2018/data_set/pre_data/output_cv.tfrecords'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


writer_cv = tf.python_io.TFRecordWriter(cv_load_dir)
writer_train = tf.python_io.TFRecordWriter(train_load_dir)

for root, dirs, files in os.walk(DIR):
    print(root, dirs, files) #当前目录路径

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

        elif len(files) == 5:
            for file in files:
                print(file)
                (filename, extension) = os.path.splitext(file)

                img = nib.load(root + '/' + file)
                img_data = img.get_fdata()
                #img_data = img_data.tolist()

                if '_seg' in filename:
                    img_data = img_data.astype(int)
                    img_data = np.where(img_data <= 3, img_data, 3)
                    """
                    for i in range(240):
                        for j in range(240):
                            for k in range(155):
                                if img_data[i, j, k] == 4:
                                    img_data[i, j, k] = 3
                    """
                    label = img_data.tobytes()
                else:
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

            if 'HGG' in root:
                img_type = b'HGG'
            if 'LGG' in root:
                img_type = b'LGG'

            example = tf.train.Example(features=tf.train.Features(feature={
                'name': _bytes_feature(patient_name),
                'image_type': _bytes_feature(img_type),
                'label': _bytes_feature(label),
                'flair': _bytes_feature(flair_img),
                'T1': _bytes_feature(t1_img),
                'T1ce': _bytes_feature(t1ce_img),
                'T2': _bytes_feature(t2_img)
            }))

            chance = np.random.randint(100)
            if chance < 10:
                data_type = b'CVset'
                writer_cv.write(example.SerializeToString())
            else:
                data_type = b'trainset'
                writer_train.write(example.SerializeToString())

            patient_name = None
            data_type = None
            img_type = None
            label = None
            flair_img = None
            t1_img = None
            t1ce_img = None
            t2_img = None
            img_max = None
            img_min = None

        print('----------------------------------')
print('Load success')
writer_cv.close()
writer_train.close()






