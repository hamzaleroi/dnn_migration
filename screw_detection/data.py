import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import cv2
from PIL import Image as imgop
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook 
from glob import glob
import imageio
import random
from tqdm.notebook import tqdm
import shutil
# standard imports
from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen,IAAAffine,JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)
 
import albumentations as albu
from albumentations import Resize
'''
# from provided source kept for reference
train_datagen = ImageDataGenerator(rescale=1./255, # no need to rescale to tfrecords
                                   rotation_range=20,width_shift_range=0.1, height_shift_range=0.1,zoom_range=[0.9,1.25],
                                   shear_range=0.01, 
                                   horizontal_flip=True,
				                           vertical_flip=True,
				                           brightness_range=[0.4,1.5],
                                   fill_mode='reflect')
'''






def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

def create_dir(base_dir,ext_name):
    new_dir=os.path.join(base_dir,ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir


def aug():
    return Compose([HorizontalFlip(p=0.5), #applied
                    VerticalFlip(p=0.5), #applied
                    ShiftScaleRotate(shift_limit=(0.1,0.1), # width_shift_range=0.1,# height_shift_range=0.1,
                                    scale_limit=(0.9,1.25),  # zoom_range=[0.9,1.25]
                                    rotate_limit=20, p=0.5), # rotation_range=20,                                                                         
                    RandomBrightnessContrast(brightness_limit=(0.4,1.5),p=0.5), # brightness_range=[0.4,1.5]
                    IAAAffine(shear=0.01,mode='reflect',p=0.5) #shear_range=0.01,fill_mode='reflect'
                    ], p = 1)




def fill_missing(source,nb_needed,iden):
  if nb_needed > 0:
    print('Filling:',iden)
    augmented=[]
    for i in tqdm(range(nb_needed)):
      img=random.choice(source)
      img=aug()(image=img)
      img=img['image']
      img=img.astype(np.uint8)
      augmented.append(img)
    return source+augmented
  else:
    return source
    
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(data,labels,save_dir,r_num):
    tfrecord_name='{}.tfrecord'.format(r_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for img,label in zip(data,labels):
            _,img_coded = cv2.imencode('.png',img)
            # Byte conversion
            image_png_bytes = img_coded.tobytes()
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'label':_int64_feature(label)
            }
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(_data,_labels,save_dir):
    for i in tqdm(range(0,len(_data),DATA_NUM)):
        data   = _data[i:i+DATA_NUM]
        labels = _labels[i:i+DATA_NUM]
        r_num=i // DATA_NUM
        to_tfrecord(data,labels,save_dir,r_num)


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

DIM=(64,64) # @param
TRAIN_DATA_PER_CLASS=51200 # @param
EVAL_DATA=5120 # @param
class_names=['non_screw','screw']
SRC_DIR=args.data_location
DATA_NUM=2048
NEEDED_DATA=[]
DATA_LIST=[]


TFIDEN = 'ScrewDTF'
tf_dir=create_dir(os.getcwd(),TFIDEN)
tf_train=create_dir(tf_dir,'Train')
tf_eval=create_dir(tf_dir,'Eval')



if __name__ == '__main__':
  for class_name in class_names:
    # class h5
    h5path=os.path.join(SRC_DIR,f"{class_name}.h5")
    # class data
    class_data=list(readh5(h5path))
    DATA_LIST.append(class_data)
    # needed data
    needed_data=TRAIN_DATA_PER_CLASS-len(class_data)
    NEEDED_DATA.append(needed_data)
    print('Class_name:{}    Found Data:{}   Needed:{}'.format(class_name,
                                                              len(class_data),
                                                                  needed_data))
  _DATA=[]
  _LABELS=[]
  for class_data,class_name,needed_data,idx in zip(DATA_LIST,class_names,NEEDED_DATA,range(len(class_names))):
    class_data=fill_missing(class_data,needed_data,class_name)
    _DATA+=class_data
    _labels=[idx for _ in range(len(class_data))]
    _LABELS+=_labels


  _comb = list(zip(_DATA,_LABELS))
  random.shuffle(_comb)
  _data, _labels = zip(*_comb)

  eval_data=_data[:EVAL_DATA]
  eval_labels=_labels[:EVAL_DATA]
  train_data=_data[EVAL_DATA:]
  train_labels=_labels[EVAL_DATA:]

  print('Creating training tfrecords')
  genTFRecords(train_data, train_labels,tf_train)
  print('Creating eval tfrecords')
  genTFRecords(eval_data, eval_labels,tf_eval)
  shutil.rmtree(os.path.join(os.getcwd(),TFIDEN))
