#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm_notebook
import shutil
import tensorflow as tf
from tqdm.notebook import tqdm
from albumentations import Resize
import albumentations as albu
from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, IAAAffine, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma, Rotate,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)
import imageio
from glob import glob
from sklearn.utils import shuffle
import random
from PIL import Image as imgop
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import argparse

TFIDEN = 'ScrewCTF'

def readh5(d_path):
    data = h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data


def create_dir(base_dir, ext_name):
    new_dir = os.path.join(base_dir, ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def aug():
    return Compose([HorizontalFlip(p=0.5),  # applied
                    VerticalFlip(p=0.5),  # applied
                    ShiftScaleRotate(shift_limit=(0.1, 0.1),  # width_shift_range=0.1,# height_shift_range=0.1,
                                     # zoom_range=[0.9,1.25]
                                     scale_limit=(0.9, 1.25),
                                     rotate_limit=20, p=0.5),  # rotation_range=20,
                    RandomBrightnessContrast(brightness_limit=(
                        0.4, 1.5), p=0.5),  # brightness_range=[0.4,1.5]
                    # shear_range=0.01,fill_mode='reflect'
                    IAAAffine(shear=0.01, mode='reflect', p=0.5)
                    ], p=1)

def fill_missing(source, nb_needed, iden):
    if nb_needed > 0:
        print('Filling:', iden)
        augmented = []
        for i in tqdm(range(nb_needed)):
            img = random.choice(source)
            img = aug()(image=img)
            img = img['image']
            img = img.astype(np.uint8)
            augmented.append(img)
        return source + augmented
    else:
        return source


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(data, labels, save_dir, r_num):
    tfrecord_name = '{}.tfrecord'.format(r_num)
    tfrecord_path = os.path.join(save_dir, tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for img, label in zip(data, labels):
            _, img_coded = cv2.imencode('.png', img)
            # Byte conversion
            image_png_bytes = img_coded.tobytes()
            data = {'image': _bytes_feature(image_png_bytes),
                    'label': _int64_feature(label)
                    }
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def genTFRecords(_data, _labels, save_dir):
    for i in tqdm(range(0, len(_data), DATA_NUM)):
        data = _data[i:i + DATA_NUM]
        labels = _labels[i:i + DATA_NUM]
        r_num = i // DATA_NUM
        to_tfrecord(data, labels, save_dir, r_num)


def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_location',  type = str, default=os.path.join(os.getcwd(),'data'), help = 'The h5 filed location')

    in_args = parser.parse_args()

    return in_args
args = get_input_args()

DIM = (75, 75)  # @param
TRAIN_DATA_PER_CLASS = 10240  # @param
root = args.data_location if os.path.exists(args.data_location) else os.path.join(root, 'data')
TRAIN_DIR = os.path.join(root, 'train')
TEST_DIR = os.path.join(root, 'test')

class_names = ['ph1',
               'slotted6.5',
               'torx7',
               'allen2.75',
               'ph2',
               'allen4',
               'torx8',
               'slotted4.5',
               'torx9',
               'torx6',
               'slotted10',
               'allen2.5']




NEEDED_DATA = []
DATA_LIST = []
# training data
for class_name in class_names:
    # class h5
    try:
        h5path = os.path.join(TRAIN_DIR, f"{class_name}.h5")
        # class data
        class_data = list(readh5(h5path))
        DATA_LIST.append(class_data)
        # needed data
        needed_data = TRAIN_DATA_PER_CLASS - len(class_data)
        NEEDED_DATA.append(needed_data)
        print('Class_name:{}    Found Data:{}   Needed:{}'.format(class_name,
                                                                len(class_data),
                                                                needed_data))
    except:
        continue


# record dir
tf_dir = create_dir(os.getcwd(), TFIDEN)
tf_train = create_dir(tf_dir, 'Train')
tf_eval = create_dir(tf_dir, 'Eval')



_DATA = []
_LABELS = []
for class_data, class_name, needed_data, idx in zip(
        DATA_LIST, class_names, NEEDED_DATA, range(len(class_names))):
    class_data = fill_missing(class_data, needed_data, class_name)
    _DATA += class_data
    _labels = [idx for _ in range(len(class_data))]
    _LABELS += _labels

_comb = list(zip(_DATA, _LABELS))
random.shuffle(_comb)
Training_data, Training_labels = zip(*_comb)



Testing_data = []
Testing_labels = []
# testing data
for class_name in tqdm(class_names):
    # class h5
    h5path = os.path.join(TEST_DIR, f"{class_name}.h5")
    # class data
    class_data = list(readh5(h5path))
    Testing_data += class_data
    labels = [class_names.index(class_name) for _ in range(len(class_data))]
    Testing_labels += labels

_comb = list(zip(Testing_data, Testing_labels))
random.shuffle(_comb)
Testing_data, Testing_labels = zip(*_comb)


DATA_NUM = 2048  # @param

# train Data
print('Creating training tfrecords')
genTFRecords(Training_data, Training_labels, tf_train)
# eval
print('Creating eval tfrecords')
genTFRecords(Testing_data, Testing_labels, tf_eval)

