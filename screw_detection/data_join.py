#!/usr/bin/env python
import h5py 
import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import shutil
import argparse

def create_dir(_path):
    if not os.path.exists(_path):
        os.mkdir(_path)
def merge_dir(_dir):
    class_names=os.listdir(_dir)
    print('Found Classes:',class_names)

    global count
    for class_name in tqdm(class_names):
        if class_name=='non_screw':
            class_dir=os.path.join(ds_dir,class_name)
            create_dir(class_dir)
        else:
            class_dir=os.path.join(ds_dir,'screw')
            create_dir(class_dir)
            
        source_dir=os.path.join(_dir,class_name)
        
        for src_ipath in tqdm(glob(os.path.join(source_dir,'*.jpg'))):
            dest_ipath=os.path.join(class_dir,f"{count}.jpg")
            shutil.copyfile(src_ipath,dest_ipath)
            count+=1
        for src_ipath in tqdm(glob(os.path.join(source_dir,'*.png'))):
            dest_ipath=os.path.join(class_dir,f"{count}.png")
            print('copying {} to {} '.format(src_ipath,dest_ipath))
            shutil.copyfile(src_ipath,dest_ipath)
            count+=1



def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--src_dir',  type = str, default=os.path.join(os.getcwd(),'data'),help = 'The folder in which images are stored')

    parser.add_argument('--dest_dir',  type = str, default=os.path.join(os.getcwd(),'comb'),help = 'The folder to chich images will be put')


    in_args = parser.parse_args()

    return in_args

if __name__ == '__main__':
    args = get_input_args()
    data_dir = args.src_dir if os.path.exists(args.src_dir) else os.path.join(os.getcwd(),'dataset')
    class_names = os.listdir(data_dir)
    print('Found Classes:',class_names)
    ds_dir=args.dest_dir if os.path.exists(args.dest_dir) else os.path.join(os.getcwd(),'data')
    create_dir(ds_dir)
    count=0
    merge_dir(data_dir)




