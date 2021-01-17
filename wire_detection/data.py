

from itertools import combinations
import tensorflow as tf
import argparse
from albumentations import Resize
import albumentations as albu
from albumentations import (
    Blur,
    Compose,
    HorizontalFlip,
    HueSaturationValue,
    IAAEmboss,
    IAASharpen,
    JpegCompression,
    OneOf,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomContrast,
    RandomCrop,
    RandomGamma,
    RandomRotate90,
    RGBShift,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion)
import imageio
from sklearn.utils import shuffle
from PIL import Image as imgop
import random
import math
from glob import glob
from skimage.morphology import disk, dilation, erosion
from skimage import io
from skimage.draw import polygon, polygon_perimeter
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook
import json
import numpy as np
import shutil
import os


def draw_contour(img_shape, x, y, channel):
    mask = np.zeros(img_shape, dtype=np.uint8)
    x = np.array(x)
    x = [min(i, img_shape[1]) for i in x]
    y = np.array(y)
    y = [min(i, img_shape[0]) for i in y]
    width, height = max(x) - min(x), max(y) - min(y)
    sq = width * height
    mask_x, mask_y = polygon(x, y)
    mask[mask_y, mask_x, channel] = 255
    return mask


def draw_contours_above(mask, x, y, channel=3):
    selen = disk(2)
    x = np.array(x)
    y = np.array(y)
    cnt_x, cnt_y = polygon_perimeter(x, y)
    mask[cnt_y, cnt_x, channel] = 255
    mask[:, :, 2] = dilation(mask[:, :, 2], selen)
    return mask


def create_dir(base_dir, ext_name):
    '''
        creates a new dir with ext_name in base_dir and returns the path
    '''
    new_dir = os.path.join(base_dir, ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def populate(_mpaths, _img, _msk, mode):
    print(f'{mode} population')
    for _mpath in tqdm_notebook(_mpaths):
        _ipath = str(_mpath).replace('raw_masks', 'raw')
        
        img_src = _ipath
        img_dst = os.path.join(_img, os.path.basename(_ipath))
        shutil.copy(img_src, img_dst)
        
        msk_src = _mpath
        msk_dst = os.path.join(_msk, os.path.basename(_mpath))
        shutil.copy(msk_src, msk_dst)


def aug():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.01,
            scale_limit=0.04,
            rotate_limit=0,
            p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        IAAEmboss(p=0.25),
        Blur(p=0.01, blur_limit=3),
        OneOf([
            ElasticTransform(
                p=0.5,
                alpha=120,
                sigma=120 *
                0.05,
                alpha_affine=120 *
                0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8)
    ], p=1)








def read_image_mask(image_path, mask_path, image_size=256):
    
    x = np.array(
        (imgop.open(image_path)).resize(
            (image_size, image_size))).astype(
        np.uint8)
    
    y = cv2.imread(mask_path, 0)
    y = cv2.resize(y, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    y = cv2.GaussianBlur(y, (5, 5), 0)
    _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    y = np.array(y).astype(np.uint8)
    return x, y




def create_data(srcp, imgp, maskp, image_size=256):
    global DATA_IDEN
    for _path in tqdm_notebook(glob(os.path.join(srcp, 'images', '*.*'))):
        _mpath = str(_path).replace('images', 'masks')
        x_sample, y_sample = read_image_mask(_path, _mpath)
        
        for _ in range(NB_AUGMIX):
            augmented = aug()(image=x_sample, mask=y_sample)
            img = augmented['image']
            tgt = augmented['mask'].reshape(image_size, image_size)
            img = img.astype(np.uint8)
            tgt = tgt.astype(np.uint8)
            imageio.imsave(os.path.join(imgp, '{}.png'.format(DATA_IDEN)), img)
            imageio.imsave(
                os.path.join(
                    maskp,
                    '{}.png'.format(DATA_IDEN)),
                tgt)
            DATA_IDEN += 1









def FlipData(img, gt, fid):
    '''
    TAKES NUMPY ARRAY
    '''
    if fid == 0:  
        x = np.array(img)
        y = np.array(gt)
    elif fid == 1:  
        x = np.array(imgop.fromarray(img).transpose(imgop.FLIP_LEFT_RIGHT))
        y = np.array(imgop.fromarray(gt).transpose(imgop.FLIP_LEFT_RIGHT))
    elif fid == 2:  
        x = np.array(imgop.fromarray(img).transpose(imgop.FLIP_TOP_BOTTOM))
        y = np.array(imgop.fromarray(gt).transpose(imgop.FLIP_TOP_BOTTOM))
    else:  
        x = imgop.fromarray(img).transpose(imgop.FLIP_TOP_BOTTOM)
        x = np.array(x.transpose(imgop.FLIP_LEFT_RIGHT))
        y = imgop.fromarray(gt).transpose(imgop.FLIP_TOP_BOTTOM)
        y = np.array(y.transpose(imgop.FLIP_LEFT_RIGHT))
    return x, y




def saveTransposedData(img, gt, imgp, maskp, comb_flag=False):
    '''
    TAKES NUMPY ARRAY
    '''
    global DATA_IDEN
    if comb_flag:
        x, y = FlipData(img, gt, random.randint(0, 4))
        rot_angle = random.randint(0, 90)
        x = np.array(imgop.fromarray(x).rotate(rot_angle))
        y = np.array(imgop.fromarray(y).rotate(rot_angle))
        fname = '{}.png'.format(DATA_IDEN)
        imageio.imsave(os.path.join(imgp, fname), x)
        imageio.imsave(os.path.join(maskp, fname), y)
        DATA_IDEN += 1
    else:
        for fid in range(4):
            x, y = FlipData(img, gt, fid)
            fname = '{}.png'.format(DATA_IDEN)
            imageio.imsave(os.path.join(imgp, fname), x)
            imageio.imsave(os.path.join(maskp, fname), y)
            DATA_IDEN += 1




def createCropAug(srcp, imgp, maskp, image_size=256):
    for img_path in tqdm_notebook(glob(os.path.join(srcp, 'images', '*.*'))):
        
        gt_path = str(img_path).replace("images", "masks")
        
        x, y = read_image_mask(img_path, gt_path)
        
        IMG = imgop.fromarray(x)
        GT = imgop.fromarray(y)
        _height, _width = image_size, image_size
        
        for pxv in [0, _width // 2, 'AC']:
            for pxl in [0, _height // 2, 'AC']:
                if (pxv != 'AC' and pxl != 'AC'):
                    left = pxv
                    right = pxv + _width // 2
                    top = pxl
                    bottom = pxl + _height // 2
                    bbox = (left, top, right, bottom)
                    _IMG = IMG.crop(bbox).resize((image_size, image_size))
                    _GT = GT.crop(bbox).resize((image_size, image_size))

                elif (pxv == 'AC' and pxl != 'AC'):
                    continue
                elif (pxl == 'AC' and pxv != 'AC'):
                    continue
                elif (pxv == 'AC' and pxl == 'AC'):
                    _GT = GT
                    _IMG = IMG
                else:
                    continue
                
                for rot_angle in [
                        0, random.randint(0, 90),
                        random.randint(0, 90),
                        random.randint(0, 90)]:  
                    rot_img = _IMG.rotate(rot_angle)
                    rot_gt = _GT.rotate(rot_angle)
                    
                    saveTransposedData(
                        np.array(rot_img), np.array(rot_gt), imgp, maskp)












def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)



def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)




def createDatafromComb(comb, img_paths, _dpath, image_size=256):
    '''
        image collage from 4 unique images that works as a completely new image
    '''
    
    _dim = image_size // 2
    bbox = [(0, 0, _dim, _dim),
            (0, _dim, _dim, image_size),
            (_dim, 0, image_size, _dim),
            (_dim, _dim, image_size, image_size)]
    
    X = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    Y = np.zeros((image_size, image_size), dtype=np.uint8)
    
    for i in range(4):
        _ipath = img_paths[comb[i]]
        _mpath = str(img_paths[comb[i]]).replace('images', 'masks')
        x, y = read_image_mask(_ipath, _mpath)
        x = np.array(imgop.fromarray(x).crop(bbox[i]))
        y = np.array(imgop.fromarray(y).crop(bbox[i]))
        X[bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3]] = (
            x * 255).astype(np.uint8)
        Y[bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3]] = (
            y * 255).astype(np.uint8)

    X = (X * 255).astype(np.uint8)
    Y = (Y * 255).astype(np.uint8)
    
    saveTransposedData(
        X, Y, os.path.join(
            _dpath, 'images'), os.path.join(
            _dpath, 'masks'), comb_flag=True)




def createCombData(srcp, _dpath, mode):
    if mode == 'train':
        nb_comb = (NB_TRAIN_REQ - NB_TRAIN * (NB_AUGMIX + NB_MANAUG))
    elif mode == 'eval':
        nb_comb = (NB_EVAL_REQ - NB_EVAL * (NB_AUGMIX + NB_MANAUG))
    img_paths = [_path for _path in glob(os.path.join(srcp, 'images', '*.*'))]
    random.shuffle(img_paths)
    vals = [i for i in range(len(img_paths))]

    if nb_comb > 0:
        for _ in tqdm_notebook(range(nb_comb)):
            comb = random_combination(vals, 4)  
            createDatafromComb(comb, img_paths, _dpath)
    else:
        print('Sufficient Data Available')



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(image_paths, save_dir, r_num):
    tfrecord_name = '{}.tfrecord'.format(r_num)
    tfrecord_path = os.path.join(save_dir, tfrecord_name)
    print(tfrecord_path)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image_path in tqdm_notebook(image_paths):
            target_path = str(image_path).replace('images', 'masks')
            with(open(image_path, 'rb')) as fid:
                image_png_bytes = fid.read()
            with(open(target_path, 'rb')) as fid:
                target_png_bytes = fid.read()
            data = {'image': _bytes_feature(image_png_bytes),
                    'target': _bytes_feature(target_png_bytes)
                    }
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def genTFRecords(data_path, mode_dir):
    data_dir = os.path.join(data_path, 'images')
    __paths = [os.path.join(data_dir, _file)
               for _file in os.listdir(data_dir)
               if os.path.isfile(os.path.join(data_dir, _file))]

    random.shuffle(__paths)
    for i in range(0, len(__paths), DATA_NUM):
        image_paths = __paths[i:i + DATA_NUM]
        random.shuffle(image_paths)
        r_num = i // DATA_NUM
        if len(image_paths) == DATA_NUM:
            to_tfrecord(image_paths, mode_dir, r_num)



def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--src_dir',  type = str, default=os.path.join(os.getcwd(),'source'),help = 'The folder in which images are stored')

    parser.add_argument('--dest_dir',  type = str, default=os.getcwd(),help = 'The folder in which images are stored')



    in_args = parser.parse_args()

    return in_args

if __name__ == '__main__':
    args = get_input_args()
    if os.path.exists(args.src_dir):
        src_dir = args.src_dir
        print(f'processing {src_dir}')
    else:
        src_dir = os.path.join(os.getcwd(), 'source')
        print(f'processing {src_dir}')
        
    json_path = os.path.join(src_dir, 'annotation.json')
    src_img_path = os.path.join(src_dir, 'raw')

    with open(json_path, 'r') as f:
        meta = json.load(f)
    masks = []
    filenames = []
    missed = []
    print('Found Annotations for:', len(meta), 'files')

    src_img_path = os.path.join(src_dir, 'raw')
    src_mask_path = create_dir(src_dir, 'raw_masks')
    img_paths = [i for i in glob(os.path.join(src_img_path, '*.png'))]
    print(f'Found {len(img_paths)} images')
    if os.path.exists(src_mask_path):
        print(f'Found mask paths at:{src_mask_path}')


    for file_meta in tqdm_notebook(meta):
        try:
            filename = file_meta.split('.png')[0] + '.png'
            img = cv2.imread(os.path.join(src_img_path, filename))
            
            img_height, img_weight = img.shape[:2]
            
            mask = np.zeros((img_height, img_weight, 3), dtype=np.uint8)
            
            file_contours = meta[file_meta]['regions']
            for cnt_dict in file_contours:
                cnt_x, cnt_y = cnt_dict['shape_attributes']['all_points_x'], cnt_dict['shape_attributes']['all_points_y']
                channel = 0
                mask += draw_contour(mask.shape, cnt_x, cnt_y, channel)
            if mask.sum() > 0:
                masks.append(mask)
                filenames.append(filename)
                io.imsave(os.path.join(src_mask_path, filename), mask)
            else:
                print('Missing:', filename)
                missed.append(filename)

        except Exception as e:
            missed.append(filename)
            print(f'ERROR: {e}', filename)


    print('Masks')

    print('Images')



    src_img = src_img_path
    src_msk = src_mask_path

    base_dir = create_dir(src_dir, 'base')

    train_dir = create_dir(base_dir, 'train')
    train_img = create_dir(train_dir, 'images')
    train_msk = create_dir(train_dir, 'masks')

    eval_dir = create_dir(base_dir, 'eval')
    eval_img = create_dir(eval_dir, 'images')
    eval_msk = create_dir(eval_dir, 'masks')

    test_dir = create_dir(base_dir, 'test')
    test_img = create_dir(test_dir, 'images')
    test_msk = create_dir(test_dir, 'masks')

    infer_dir = create_dir(base_dir, 'infer')



    if missed:
        print('Copying missing mask files to infer')
        for filename in tqdm_notebook(missed):
            missed_src = os.path.join(src_img, filename)
            missed_dst = os.path.join(infer_dir, filename)
            shutil.copy(missed_src, missed_dst)
    else:
        print('No file Mask Missing')


    print('Copying non-masked image files to infer')
    for _ipath in tqdm_notebook(glob(os.path.join(src_img, '*.png'))):
        if not os.path.exists(str(_ipath).replace('raw', 'raw_masks')):
            filename = os.path.basename(_ipath)
            missed_src = os.path.join(src_img, filename)
            missed_dst = os.path.join(infer_dir, filename)
            shutil.copy(missed_src, missed_dst)


    _mpaths = [i for i in glob(os.path.join(src_msk, '*.png'))]
    NB_DATA = len(_mpaths)
    NB_TRAIN = math.floor((100 / 130) * NB_DATA)
    NB_EVAL = max(10, math.floor((10 / 130) * NB_DATA))
    NB_TEST = NB_DATA - (NB_TRAIN + NB_EVAL)
    print('Data:', NB_DATA, 'Train:', NB_TRAIN, 'Eval:', NB_EVAL, 'Test:', NB_TEST)

    random.shuffle(_mpaths)
    train_mpaths = _mpaths[:NB_TRAIN]
    eval_mpaths = _mpaths[NB_TRAIN:NB_TRAIN + NB_EVAL]
    test_mpaths = _mpaths[NB_TRAIN + NB_EVAL:]
    populate(train_mpaths, train_img, train_msk, 'train')
    populate(eval_mpaths, eval_img, eval_msk, 'eval')
    populate(test_mpaths, test_img, test_msk, 'test')

    NB_AUGMIX = 20  
    NB_MANAUG = 80  
    IMG_DIM = 256  
    DATA_NUM = 1024  
    DATA_IDEN = 0   
    NB_TRAIN_REQ = 10240  
    NB_EVAL_REQ = 1024  

    if os.path.exists(args.dest_dir):
        dest_dir = args.dest_dir
        print(f'processing {src_dir}')
    else:
        dest_dir = os.getcwd()
        print(f'processing {src_dir}')


    ds_dir = create_dir(dest_dir, 'DataSet')

    ds_train_dir = create_dir(ds_dir, 'Train')
    ds_train_img = create_dir(ds_train_dir, 'images')
    ds_train_mask = create_dir(ds_train_dir, 'masks')

    ds_eval_dir = create_dir(ds_dir, 'Eval')
    ds_eval_img = create_dir(ds_eval_dir, 'images')
    ds_eval_mask = create_dir(ds_eval_dir, 'masks')

    tf_dir = create_dir(ds_dir, 'WireDTF')
    tf_train = create_dir(tf_dir, 'Train')
    tf_eval = create_dir(tf_dir, 'Eval')
 
 
    create_data(train_dir, ds_train_img, ds_train_mask)
    createCropAug(train_dir, ds_train_img, ds_train_mask)
    createCombData(train_dir, ds_train_dir, 'train')
    genTFRecords(ds_train_dir, tf_train)


    create_data(eval_dir, ds_eval_img, ds_eval_mask)
    createCropAug(eval_dir, ds_eval_img, ds_eval_mask)
    createCombData(eval_dir, ds_eval_dir, 'eval')
    genTFRecords(ds_eval_dir, tf_eval)

