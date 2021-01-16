#!/usr/bin/env python
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import tensorflow as tf
import warnings
import os
from tqdm import tqdm



def create_model(iden,NB_CLASS,NB_CHANNEL,WEIGHT_PATH):
  if iden=='densenet201':
    base_model_wrapper=tf.keras.applications.DenseNet201
    IMG_DIM=221
  if iden=='inceptionResNetv2':
    base_model_wrapper=tf.keras.applications.InceptionResNetV2
    IMG_DIM=139
  if iden=='inceptionv3':
    base_model_wrapper=tf.keras.applications.InceptionV3
    IMG_DIM=139
  if iden=='resnet101v2':
    base_model_wrapper=tf.keras.applications.ResNet101V2
    IMG_DIM=64
  if iden=='resnext101':
    base_model_wrapper=ResNeXt101
    IMG_DIM=64
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
  x = tf.keras.layers.Dense(NB_CLASS, activation='softmax')(x)
  model =tf.keras.models.Model(inputs=base_model.input,outputs=x,name=iden)
  model.load_weights(WEIGHT_PATH)
  dim=IMG_DIM
  return model,dim




def predict_on_data(_paths, export_folder='/tmp'):
  for img_path in tqdm(_paths):
    img_raw = cv2.imread(img_path)
    img_raw = img_raw[:,:,:3]
    img_h, img_w = img_raw.shape[:2]
    if img_h>img_w:
      ratiox = DIM1/img_w
      ratioy = DIM2/img_h
      img_raw = cv2.resize(img_raw, (DIM1,DIM2))
    else:
      ratiox = DIM2/img_w
      ratioy = DIM1/img_h
      img_raw = cv2.resize(img_raw, (DIM2,DIM1))
    
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                              1, 100, 
                              param1=hough_upper_threshold,
                              param2=hough_lower_threshold,
                              minRadius=hough_min_radius,
                              maxRadius=hough_max_radius)
    
    
    if circles is not None:
      # convert the (x, y) coordinates and radius of the circles to integers
      circles = np.round(circles[0, :]).astype("int")
      # copy the image, for painting we will use another
      drawn_image   = img_raw.copy()
      drawn_image   = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
      # loop over the found circles
      for i in range(len(circles)):
        # get one
        (x, y, r) = circles[i]
        # draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
        cv2.rectangle(drawn_image, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)
        # bbox
        xmin = x-r
        xmax = x+r
        ymin = y-r
        ymax = y+r
        # get the above rectangle as ROI
        screw_roi = img_raw[ymin:ymax,xmin:xmax]
        #can't go on with the empty or corrupt roi
        if (screw_roi.size == 0):
            break
            
        # integreated prediction --> same as work
        pred_val=0
        for model,dim in INTEGRATED:
          # imgae
          data = cv2.resize(screw_roi,(dim,dim))
          data = data.astype('float32')/255.0
          tensor = np.expand_dims(data,axis=0)
          pred=model.predict(tensor)[0]
          pred_val+=pred[1]
          
        score=pred_val/2
        if score>score_thresh:
          cv2.circle(drawn_image, (int(x), int(y)), int(r), (0, 255, 0), 5) #green
    cv2.imwrite(os.path.join(export_folder,'inference_'  + img_path.split('/')[-1]) if os.path.exists(export_folder) else os.path.join(os.getcwd(),'inference_'  + img_path.split('/')[-1]),drawn_image) 
# getting input arguments
def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_location',  type = str, default=os.path.join(os.getcwd(),'screw_detection','weights'), help = 'The model location')
    
    parser.add_argument('image_path', type = str, help = 'Image to apply inference on')
     
    
     


    in_args = parser.parse_args()

    return in_args

    
    
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


idens=['densenet201','inceptionResNetv2','inceptionv3','resnet101v2','resnext101','xception']
# parameters
NB_CHANNEL=3 # @param
NB_CLASS=2 # @param
INTEGRATED=[]
INFER_WHOLE_FOLDER = False #@param {type:"boolean"}

hough_upper_threshold = 70 # @param {type:"slider", min:0, max:100, step:1}
hough_lower_threshold = 10 # @param {type:"slider", min:0, max:100, step:1}
hough_min_radius = 5 # @param {type:"slider", min:0, max:100, step:1}
hough_max_radius = 20 # @param {type:"slider", min:0, max:100, step:1}
score_thresh=0.8 # @param {type:"slider", min:0, max:1.0, step:0.01}
DIM1=986 # @param 
DIM2=1382 # @param


args = get_input_args()

if __name__ == '__main__':
    
    path = args.model_location
    print(path)
    root = path if os.path.isabs(path)  else os.path.join(os.getcwd(),path.strip('./'))
    # modeling
    model1= 'inceptionv3' # @param ['densenet201','inceptionResNetv2','inceptionv3','resnet101v2','resnext101','xception']
    model2= 'xception' # @param ['densenet201','inceptionResNetv2','inceptionv3','resnet101v2','resnext101','xception']
    integrated=[model1,model2]
 

    # Weights
    WEIGHTS_INTGRATED=[os.path.join(root,'{}.h5'.format(iden)) 
                      for iden in integrated]

    
    for iden,WEIGHT_PATH in zip(integrated,WEIGHTS_INTGRATED):
      print('Loading Integrated Models:',iden)
      INTEGRATED.append(create_model(iden,NB_CLASS,NB_CHANNEL,WEIGHT_PATH))


    # # ROI and Prediction Wrappers


    
    DATA_PATH=args.image_path


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

    predict_on_data(_paths)