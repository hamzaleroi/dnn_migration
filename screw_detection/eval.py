#!/usr/bin/env python
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob 
import os 
import h5py
import tensorflow as tf
from tqdm.notebook import tqdm
from scripts.resnet import ResNeXt101
import warnings
from glob import glob
import cv2
import imageio
import argparse

def create_model(iden,NB_CLASS,NB_CHANNEL,WEIGHT_PATH):
  print(WEIGHT_PATH)
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


def data_input_fn(mode,BUFFER_SIZE,BATCH_SIZE,data_img_dim=64,DATA_PATH='ScrewDTF'): 
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'label'  : tf.io.FixedLenFeature([],tf.int64)
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=3)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(data_img_dim,data_img_dim,3))
        #image=tf.image.resize(image, [IMG_DIM,IMG_DIM])
        
        label=parsed_example['label']
        label=tf.cast(label,tf.int64)
        label=tf.one_hot(label,NB_CLASS)
        return image,label

    files_pattern=  DATA_PATH if os.path.isabs(DATA_PATH)  else os.path.join(os.getcwd(),DATA_PATH.strip('./'),mode,'*.tfrecord')
    print(files_pattern)
    file_paths = tf.io.gfile.glob(files_pattern)
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# data idens and predictions
def create_dir(_path):
  if not os.path.exists(_path):
    os.mkdir(_path)
  return _path

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
	
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    #print (rec)
    return ap

def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--eval_data',type = str, default='ScrewDTF', help = 'The model location')
        
    parser.add_argument('--saved_weights', type = str,  help = 'Load the pre-trained weights')
     


    in_args = parser.parse_args()

    return in_args




args = get_input_args()

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# modeling
idens=['densenet201','inceptionResNetv2','inceptionv3','resnet101v2','resnext101','xception']
model1= 'inceptionv3' # @param ['densenet201','inceptionResNetv2','inceptionv3','resnet101v2','resnext101','xception']
model2= 'densenet201' # @param ['densenet201','inceptionResNetv2','inceptionv3','resnet101v2','resnext101','xception']
integrated=[model1,model2]

# parameters
NB_CHANNEL=3 # @param
NB_CLASS=2 # @param
BATCH_SIZE=128 # @param
BUFFER_SIZE=2048 # @param
EVAL_DATA=2048*3 # @param
EVAL_STEPS      = EVAL_DATA//BATCH_SIZE
data_img_dim=64 # @param



WEIGHTS_INTGRATED=[(os.path.join(args.saved_weights,'{}.h5'.format(iden)  ) \
if os.path.isabs(args.saved_weights)\
 else os.path.join(os.getwd(), args.saved_weights.strip('./'),'{}.h5'.format(iden)  ))\
 if args.saved_weights != None else os.path.join(os.getcwd(),'weights','{}.h5'.format(iden))
                  for iden in integrated]
# data


if os.path.exists(args.eval_data):
    DATA_PATH=args.eval_data
else:
    print('Validation not found in the current folder, please specify it !!')
    exit(1)

eval_ds = data_input_fn("Eval",BUFFER_SIZE,BATCH_SIZE,DATA_PATH=args.eval_data)
print(eval_ds)

# # model creation



INTEGRATED=[]
for iden,WEIGHT_PATH in zip(integrated,WEIGHTS_INTGRATED):
  print('Loading Integrated Models:',iden)
  INTEGRATED.append(create_model(iden,NB_CLASS,NB_CHANNEL,WEIGHT_PATH))



tps,tns,fps,fns, acc=[0]*80,[0]*80,[0]*80,[0]*80,[]
tp,tn,fp,fn=0,0,0,0
print('Extracting Test Data from tfrecords')
screw_data=[]
non_screw_data=[]

for x_batch,y_batch in tqdm(eval_ds.take(EVAL_STEPS),total=EVAL_STEPS):
  for x,y in zip(x_batch,y_batch):
    label=np.argmax(y)
    if label==0:
      non_screw_data.append(x)
    else:
      screw_data.append(x)


# ## Screw Data 
print('Evaluating: Screw_data')
for i in tqdm(range(0,len(screw_data),BATCH_SIZE)):
  data=screw_data[i:i+BATCH_SIZE]
  if len(data)==BATCH_SIZE:
    scores=np.zeros((BATCH_SIZE,NB_CLASS))
    for model,dim in INTEGRATED:
      x_batch=[]
      for x in data:
        x=cv2.resize(np.array(x),(dim,dim))
        x_batch.append(x)
      x_batch=np.array(x_batch)
      y_batch=model.predict_on_batch(x_batch)
      scores+=y_batch
    scores=scores[:,1]
    for score in scores:
      thresh = 0.7
      step = 0.01
      for idx in range(79):
        if score>thresh:
          tps[idx] +=1
        else:
          fps[idx] +=1
        thresh+=step
      if score>0.8:
          tp +=1
      else:
          fp +=1
    



# ## Non Screw Data
print('Evaluating:Non Screw_data')
for i in tqdm(range(0,len(non_screw_data),BATCH_SIZE)):
  data=non_screw_data[i:i+BATCH_SIZE]
  if len(data)==BATCH_SIZE:
    scores=np.zeros((BATCH_SIZE,NB_CLASS))
    for model,dim in INTEGRATED:
      x_batch=[]
      for x in data:
        x=cv2.resize(np.array(x),(dim,dim))
        x_batch.append(x)
      x_batch=np.array(x_batch)
      y_batch=model.predict_on_batch(x_batch)
      scores+=y_batch
    scores=scores[:,1]
    for score in scores:
      thresh = 0.7
      step = 0.01
      for idx in range(79):
        if score>thresh:
          fns[idx] +=1
        else:
          tns[idx] +=1
        thresh+=step
      if score>0.8:
          fn +=1
      else:
          tn +=1


for i in range(79):
  try:
    accuracy = (tps[i]+tns[i])/(tps[i]+tns[i]+fps[i]+fns[i])
    acc.append(accuracy)
  except:
    raise
    pass

print('Models:',integrated[0],integrated[1])    
print('maximum accuracy: ',max(acc))

accuracy = (tp+tn)/(tp+tn+fp+fn)
print('TP: ', tp, ' TN: ', tn, ' FP: ', fp, ' FN: ', fn)
print('accuracy: ', accuracy)


# # Scenes Data and Hough Params
# hough parameters, subject to change, depending on the height, illumination and etc.
hough_upper_threshold = 100 # @param
hough_lower_threshold = 25 # @param
hough_min_radius = 15 # @param
hough_max_radius = 30 # @param
ovthresh=0.5 # @param
score_thresh=0.8 # @param

src_img_dir= os.path.join(os.getcwd(),'data','scenes')
src_gt_txt = os.path.join(os.getcwd(),'data','scenes.txt')
pred_dir=create_dir(os.path.join(os.getcwd(),'predictions'))

with open(src_gt_txt,'r') as txt:
  eval_data=[s.rstrip() for s in txt]

eval_data_idens=[data.split(' ')[0] for data in eval_data]

#read groundtruth
#groundtruth format: image+path xmin,ymin,xmax,ymax,0 xmin,ymin,xmax,ymax,0 .....
print('Creating Class Records')
npos = 0
class_recs = {}
for line in tqdm(eval_data):
  line_split = line.strip().split('.png ')
  image_id = line_split[0]
  boxes = line_split[1].split(' ')
  bbox = []
  for box in boxes:
      xmin, ymin, xmax, ymax, xx = box.split(',')
      bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
  bbox = np.array([x for x in bbox])
  difficult = np.array([0 for x in bbox]).astype(np.bool)
  det = [False] * len(bbox)
  npos = npos + sum(~difficult)
  class_recs[image_id] = {'bbox': bbox,
                            'difficult': difficult,
                            'det': det}
  


# # Saving Scenes Predictions and Evaluation

DIM1=986 
DIM2=1382
det=[]
for img_iden in tqdm(eval_data_idens):
  img_path=os.path.join(src_img_dir,img_iden)
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
      # bbox
      
      xmin = xmin/ratiox
      xmax = xmax/ratiox
      ymin = ymin/ratioy
      ymax = ymax/ratioy
      
      # predictions
      
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
      # evaluation
      line_out = img_iden.split('.')[0]
      line_out += ' ' + str(score) + ' ' + str(xmin) + ' ' + str(ymin)+' ' + str(xmax) + ' ' + str(ymax) 
      det.append(line_out)
      
  imageio.imsave(os.path.join(pred_dir,img_iden),drawn_image)
  


# # Scoring

#det format: image_id score xmin ymin xmax ymax
splitlines = [x.strip().split(' ') for x in det]
image_ids = [x[0] for x in splitlines]
confidence = np.array([float(x[1]) for x in splitlines])
BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
# sort by confidence
sorted_ind = np.argsort(-confidence)
sorted_scores = np.sort(-confidence)
BB = BB[sorted_ind, :]
image_ids = [image_ids[x] for x in sorted_ind]
# go down dets and mark TPs and FPs
nd = len(image_ids)
tp = np.zeros(nd)
fp = np.zeros(nd)

for d in range(nd):
    R = class_recs[image_ids[d]]
    bb = BB[d, :].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)

    if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    if ovmax > ovthresh:
        if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
    else:
        fp[d] = 1.

# compute precision recall
fp = np.cumsum(fp)
tp = np.cumsum(tp)
rec = tp / float(npos)
# avoid divide by zero in case the first detection matches a difficult
# ground truth
prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
ap = voc_ap(rec, prec, use_07_metric=False)

print('AP = {:.4f}'.format( ap))
