#!/usr/bin/env python
# coding: utf-8
import imageio
import numpy as np
import cv2
from PIL import Image as imgop
from glob import glob
import segmentation_models as sm
import os
import tensorflow as tf
import argparse 

def get_img(_path):
    data = imgop.open(_path)
    data = data.resize((IMG_DIM, IMG_DIM))
    data = np.array(data)
    data = data.astype('float32') / 255.0
    data = np.expand_dims(data, axis=0)
    return data


def get_pred(model, img, _path):
    pred = model.predict([img])
    pred = np.squeeze(pred) * 255.0
    pred = pred.astype('uint8')
    imageio.imsave('temp.png', pred)
    pred = cv2.imread('temp.png', 0)
    pred = cv2.resize(pred, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(pred, (5, 5), 0)
    _, pred = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    os.remove('temp.png')
    return pred


def get_overlay(pred, img):
    img = np.squeeze(img)
    msk_3d = np.zeros(img.shape)
    xs, ys = np.nonzero(pred)
    for x, y in zip(xs, ys):
        msk_3d[x, y, 0] = 1
    overlay = img * 0.5 + msk_3d * 0.5
    overlay = overlay * 255.0
    overlay = overlay.astype('uint8')
    return overlay

def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--saved_weights',  type = str, default=os.path.join(os.getcwd(),'weights'),help = 'Where the weights are saved')

    parser.add_argument('--infer_path',  type = str, default=os.getcwd(),help = 'The folder in which images are stored')



    in_args = parser.parse_args()

    return in_args

if __name__ == '__main__':
    args = get_input_args()
    model_name = 'efficientnetb7'
    iden = 'model'
    sm.set_framework('tf.keras')
    if os.path.exists(args.saved_weights):
        saved_weights = os.path.join(args.saved_weights, f'{iden}.h5')
        print(f'processing {saved_weights}')
    else:
        saved_weights = os.path.join(os.getcwd(), 'weights', f'{iden}.h5')
        print(f'processing {saved_weights}')
    IMG_DIM = 256
    NB_CHANNEL = 3
    WEIGHT_PATH = saved_weights
    
    mod = sm.Unet(
        model_name,
        input_shape=(
            IMG_DIM,
            IMG_DIM,
            NB_CHANNEL),
        encoder_weights=None)

    mod.load_weights(WEIGHT_PATH)
    print('Loaded inference weights')
    if os.path.exists(args.infer_path):
        infer_path = os.path.join(args.infer_path)
        print(f'processing {infer_path}')
    else:
        infer_path = os.path.join(os.getcwd(), 'inference')
        print(f'processing {infer_path}')
    img_dir = os.path.join(infer_path, 'images')
    pred_dir = os.path.join(infer_path, 'predictions')
    print('Found infer dir:', infer_path)
    check = len([i for i in glob(os.path.join(img_dir, '*.*'))])
    if check == 0:
        print('No images found to infer on. See the notes at the begining of the script.')

    if check != 0:
        img_paths = glob(os.path.join(img_dir, '*.*'))
        for _path in img_paths:
            # image ([])
            img = get_img(_path)
            # predictions ([])
            pred = get_pred(mod, img, _path)
            # overlays ([])
            overlay = get_overlay(pred, img)
            imageio.imsave(
                os.path.join(
                    pred_dir,
                    os.path.basename(_path)),
                overlay)
