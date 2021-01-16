#!/usr/bin/env python

import tensorflow as tf
import os 
import argparse
import warnings
import efficientnet.tfkeras as efn
from glob import glob
import numpy as np
from tqdm.notebook import tqdm
import cv2




class COLORS:
  red   = (0,0,255)
  blue   = (255,0,0)
  green = (0,255,0)

def predict_on_data(_paths,export_folder='/tmp',**hough_params):
  for img_path in tqdm(_paths):
    print(img_path)
    img = cv2.imread(img_path)
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    circles = detect_circles(img, dp=1, **hough_params)
    if circles is not None:
      imgs = cut_rois(img, circles)
      preds=[]
      pred_circles=[]
      for roi,circle in zip(imgs,circles):
        
        '''
        roi1=cv2.resize(roi,(DET_DIM1,DET_DIM1))
        roi1=roi1.astype('float32')/255.0
        roi1=np.expand_dims(roi1,axis=0)
        det1=det_model1.predict([roi1])[0]
        '''
        roi2=cv2.resize(roi,(DET_DIM2,DET_DIM2))
        roi2=roi2.astype('float32')/255.0
        roi2=np.expand_dims(roi2,axis=0)
        det2=det_model2.predict([roi2])[0]

        if det2[1] >0.9:
            #plt.imshow(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
            #plt.show()
            img_roi=cv2.resize(roi,(IMG_DIM,IMG_DIM))
            img_roi=np.expand_dims(img_roi,axis=0)
            img_roi=img_roi.astype('float32')/255.0
            idx=np.argmax(model.predict(img_roi)[0])
            preds.append(class_names[idx])
            pred_circles.append(circle)
        else:
            preds.append('non_screw')
   
      pim = draw_preds(img, preds, circles)
      final=draw_circles(pim, pred_circles)
      final=cv2.cvtColor(final,cv2.COLOR_BGR2RGB)

      path = os.path.join(export_folder,
        'inference_'  + img_path.split('/')[-1]) if os.path.exists(export_folder) else os.path.join(os.getcwd(),
        'inference_'  + img_path.split('/')[-1])
      print(f'SAVING {path} ...')
      cv2.imwrite(path,final) 
      


def detect_circles(im, **kwargs):
  x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  x = cv2.resize(x, (0,0), fx=.3, fy=.3)
  h_circles = cv2.HoughCircles(x, cv2.HOUGH_GRADIENT, **kwargs)
  circles = h_circles[0] if h_circles is not None else None
  if circles is not None:
    return (circles/.3).astype(int)
  else:
    return None

def draw_circles(im, circles):
  out = im.copy()
  for x,y,r in circles:  cv2.circle(out, (x,y), r, COLORS.green, 5)
  return out

def draw_preds(im, preds, circles, sz=2, thick=4, color=COLORS.green):
  im = im.copy()
  for (x,y,r),p in zip(circles, preds):
    if p!='non_screw':
      cv2.putText(im, p, (x+r,y+r), cv2.FONT_HERSHEY_COMPLEX, sz, color, thick, cv2.LINE_AA)
  return im

def cut_rois(im, circles):
  rois = []
  y_max,x_max,_ = im.shape
  for x,y,r in circles:
    up,down    = max(y-r,0),min(y+r,y_max)
    left,right = max(x-r,0),min(x+r,x_max)
    rois.append(im[up:down,left:right].copy())
  return rois



def create_det_model(iden,NB_CHANNEL,WEIGHT_PATH):
  if iden=='inceptionv3':
    base_model_wrapper=tf.keras.applications.InceptionV3
    IMG_DIM=139
  if iden=='xception':
    base_model_wrapper=tf.keras.applications.Xception
    IMG_DIM=71 


  base_model = base_model_wrapper(include_top=False,
                                  weights=None,
                                  input_shape=(IMG_DIM,IMG_DIM,NB_CHANNEL))
  for layer in base_model.layers:
    layer.trainable = True
  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = tf.keras.layers.Dense(2, activation='softmax')(x)
  model =tf.keras.models.Model(inputs=base_model.input,outputs=x,name=iden)
  model.load_weights(WEIGHT_PATH)
  return model,IMG_DIM


def create_model(IMG_DIM,NB_CHANNEL,WEIGHT_PATH,NB_CLASS):
  base_model_wrapper=efn.EfficientNetB2
  base_model = base_model_wrapper(include_top=False,
                                  weights=None,
                                  input_shape=(IMG_DIM,IMG_DIM,NB_CHANNEL))
  for layer in base_model.layers:
    layer.trainable = True
  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = tf.keras.layers.Dense(NB_CLASS, activation='softmax')(x)
  model =tf.keras.models.Model(inputs=base_model.input,outputs=x,name=iden)
  model.load_weights(WEIGHT_PATH)
  return model


def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--classification_model_location',  type = str, help = 'The classification model location')
    
    parser.add_argument('--detection_model_location',  type = str, help = 'The detection model location')
    
    parser.add_argument('image_path', type = str, help = 'Image to apply inference on')
    
    parser.add_argument('--infer_folder', type = bool, default=False, help = 'Apply inference on a complete folder')
     
    
     


    in_args = parser.parse_args()

    return in_args


warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

args = get_input_args()

iden='efficientnetb2' 
det_iden1='inceptionv3'
det_iden2='xception'

IMG_DIM=256 # @param

class_names=['ph1', 
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

NB_CLASS=len(class_names)
NB_CHANNEL=3 # @param



if __name__ == '__main__':

    if args.classification_model_location != None:
        path = args.classification_model_location
       
        if os.path.exists(path) and os.path.isdir(path) :
            print('path_classification',path)
            WEIGHT_PATH= os.path.join(path,f"{iden}.h5") if os.path.isabs(path)            else  os.path.join(os.getcwd(),path.strip('./'),f"{iden}.h5")
        elif os.path.exists(''.join(path.split('/')[:-1])):
            WEIGHT_PATH= path if os.path.isabs(path)  else os.path.join(os.getcwd(),path.strip('./'))
        else:
            WEIGHT_PATH= os.path.join(os.getcwd(),'weights',f"{iden}.h5")
            print(f'wrong link saving to {SAVE_PATH}')
    else:
        WEIGHT_PATH= os.path.join(os.getcwd(),'weights',f"{iden}.h5")

    if args.detection_model_location != None:
        path = args.detection_model_location
        print('path_detection',path)
        if os.path.exists(path) and os.path.isdir(path) :
            WEIGHT_PATH_DET2= os.path.join(path,f"{det_iden2}.h5") if os.path.isabs(path)            else  os.path.join(os.getcwd(),path.strip('./'),f"{det_iden2}.h5")
        elif os.path.exists(''.join(path.split('/')[:-1])):
            WEIGHT_PATH_DET2= path if os.path.isabs(path)  else os.path.join(os.getcwd(),path.strip('./'))
        else:
            WEIGHT_PATH_DET2= os.path.join(os.getcwd(),'weights',f"{det_iden2}.h5")
            print(f'wrong link saving to {SAVE_PATH}')
    else:
        WEIGHT_PATH_DET2= os.path.join(os.getcwd(),'weights',f"{det_iden2}.h5")

    print(WEIGHT_PATH,WEIGHT_PATH_DET2)
    model=create_model(IMG_DIM,NB_CHANNEL,WEIGHT_PATH,NB_CLASS)
    det_model2,DET_DIM2=create_det_model(det_iden2,NB_CHANNEL,WEIGHT_PATH_DET2)

    print('Classification Model:',iden)
    print('Detection Model 2:',det_iden2)


    INFER_WHOLE_FOLDER = args.infer_folder #@param {type:"boolean"}
    DATA_PATH=args.image_path 
    print('data_path',DATA_PATH)

    if INFER_WHOLE_FOLDER:
      if '.' in DATA_PATH:
        print('Please Provide a folder path')
      else:
        _paths=[_path for _path in glob(os.path.join(DATA_PATH,'*.*'))]
        print('Found Images:')
        for _path in _paths:
          print(_path)
    else:
      if os.path.isfile(DATA_PATH):
        _paths=[DATA_PATH]
        print('Found Image:')
        print(_paths[0])
      else:
        print('The provided DATA_PATH is Not a file')


    # # hough Params
    hough_upper_threshold = 100 # @param {type:"slider", min:0, max:100, step:1}
    hough_lower_threshold = 30 # @param {type:"slider", min:0, max:100, step:1}
    hough_min_radius = 5 # @param {type:"slider", min:0, max:100, step:1}
    hough_max_radius = 40 # @param {type:"slider", min:0, max:100, step:1}
    hough_params = dict(minDist=100, 
                        param1=hough_upper_threshold, 
                        param2=hough_lower_threshold,
                        minRadius=hough_min_radius,
                        maxRadius=hough_max_radius)




    print(hough_params)
    predict_on_data(_paths, **hough_params)

