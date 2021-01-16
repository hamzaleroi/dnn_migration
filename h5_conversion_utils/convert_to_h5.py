import h5py 
import cv2
import argparse
import numpy as np
import os
import random
from glob import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import h5py 
import cv2
import numpy as np
import os
import shutil
from glob import glob
from tqdm.notebook import tqdm


def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',data=data)
    hf.close()

def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

def save(data,iden):
    data=np.array(data)
    h5path=os.path.join(os.getcwd(),f"{iden}.h5")
    saveh5(h5path,data)

def create_h5_data(_dir,iden):
    print('Creating Data Store:',iden)
    data=[]
    for img_path in tqdm(glob(os.path.join(_dir,'*.*'))):
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        data.append(img)
    save(data,os.path.join(_dir,iden))

def create_dir(_path):
    if not os.path.exists(_path):
        os.mkdir(_path)


def merge_dir(_dir):
    class_names = os.listdir(_dir)
    print('Found Classes:',class_names)

    global count
    for class_name in tqdm(class_names):
        if class_name=='non_screw':
            class_dir=os.path.join(data_dir,class_name)
            create_dir(class_dir)
        else:
            class_dir=os.path.join(data_dir,'screw')
            create_dir(class_dir)
            
        source_dir=os.path.join(_dir,class_name)
        
        for src_ipath in tqdm(glob(os.path.join(source_dir,'*.jpg'))):
            dest_ipath=os.path.join(class_dir,f"{count}.jpg")
            shutil.copy(src_ipath,dest_ipath)
            count+=1
        for src_ipath in tqdm(glob(os.path.join(source_dir,'*.png'))):
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
    
    parser.add_argument('--image_folder',  type = str,default=os.path.join(os.getcwd(),'dataset'),
     help = 'The folder containing the images, where:\n screws are put in a folder called according to their class name \n and any other image is put in a folder called non_screw')
    
    parser.add_argument('--dest_folder',  type = str, default=os.path.join(os.getcwd(),'data'),help = 'Where to save data')

    parser.add_argument('--delete_data',  type = bool, default=False,help = 'Where to save data')

    in_args = parser.parse_args()

    return in_args

args = get_input_args()

if __name__ == '__main__':
    count=0
    ds_dir = args.dest_folder if os.path.exists(args.dest_folder) else os.path.join(os.getcwd(),'data')
    data_dir= os.path.join(args.image_folder,'data') if os.path.exists(args.image_folder) else os.path.join(os.getcwd(),'data')
    create_dir(data_dir)
    merge_dir(ds_dir)
    if args.delete_data:
        shutil.rmtree(ds_dir)

    # fixed params
    dim=(64,64) #based on minimum dimension of the models

    class_names=['non_screw', 'screw']
    # helpers        
    for class_name in class_names:
        class_dir=os.path.join(data_dir,class_name)
        create_h5_data(class_dir,class_name)
