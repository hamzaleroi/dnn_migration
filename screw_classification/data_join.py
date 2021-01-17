#!/usr/bin/env python
import h5py 
import cv2
import argparse
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import shutil






def create_dir(_path):
    if not os.path.exists(_path):
        os.mkdir(_path)

def merge_dir(_dir):
    global count
    for class_name in tqdm(class_names):
        class_dir=os.path.join(train_dir,class_name)
        create_dir(class_dir)
        source_dir=os.path.join(_dir,class_name)
        for src_ipath in glob(os.path.join(source_dir,'*.jpg')):
            dest_ipath=os.path.join(class_dir,f"{count}.jpg")
            shutil.copy(src_ipath,dest_ipath)
            count+=1
        for src_ipath in glob(os.path.join(source_dir,'*.png')):
            dest_ipath=os.path.join(class_dir,f"{count}.png")
            shutil.copy(src_ipath,dest_ipath)
            count+=1

def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--work_dir',  type = str, default=os.getcwd(),help = 'Where the folders old and new')


    in_args = parser.parse_args()

    return in_args


if __name__ == '__main__':
    args = get_input_args()
    work_dir = args.work_dir if os.path.exists(args.work_dir) else os.getcwd()
    new_dir = os.path.join(work_dir,'new')
    old_dir = os.path.join(work_dir,'old')
    class_names=os.listdir(old_dir)
    print('Found Classes:',class_names)
    train_dir=os.path.join(work_dir,'comb')
    create_dir(train_dir)
    count=0
    merge_dir(old_dir)
    merge_dir(new_dir)

