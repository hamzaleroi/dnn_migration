
from tqdm.notebook import tqdm
from glob import glob 
import os 
import argparse
import tensorflow as tf
import numpy as np 
from scripts.resnet import ResNeXt101



def data_input_fn(mode,BUFFER_SIZE,BATCH_SIZE,IMG_DIM,data_img_dim=64,DATA_PATH='ScrewDTF'): 
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'label'  : tf.io.FixedLenFeature([],tf.int64)
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=3)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(data_img_dim,data_img_dim,3))
        image=tf.image.resize(image, [IMG_DIM,IMG_DIM])
        
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

def create_model(IMG_DIM,NB_CLASS,NB_CHANNEL):
  base_model = base_model_wrapper(include_top=False,
                                  weights=TRANSFER_LEARNING,
                                  input_shape=(IMG_DIM,IMG_DIM,NB_CHANNEL))
  for layer in base_model.layers:
    layer.trainable = True
  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = tf.keras.layers.Dense(NB_CLASS, activation='softmax')(x)
  model =tf.keras.models.Model(inputs=base_model.input,outputs=x,name=iden)
  return model


def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training_data',type = str, default='ScrewDTF', help = 'The model location')
    
    parser.add_argument('--save_location', type = str, help = 'Where to save the trained model')
    
    parser.add_argument('--saved_weights', type = str,  help = 'Load the pre-trained weights')
     
    parser.add_argument('--batch_size', type = int, default=128, help = 'batch_size')
     


    in_args = parser.parse_args()

    return in_args




args = get_input_args()
iden='xception' # @param ['densenet201','inceptionResNetv2','inceptionv3','resnet101v2','resnext101','xception']
NB_CHANNEL=3 # @param
BATCH_SIZE= args.batch_size # @param
BUFFER_SIZE=2048 # @param
TRAIN_DATA=2048*48 # @param
EVAL_DATA=2048*3 # @param
EPOCHS=250 # @param
NB_CLASS=2 # @param
TOTAL_DATA=TRAIN_DATA+EVAL_DATA
STEPS_PER_EPOCH = TOTAL_DATA//BATCH_SIZE
EVAL_STEPS      = EVAL_DATA//BATCH_SIZE
data_img_dim=64 # @param
WEIGHT_PATH= args.saved_weights if args.saved_weights != None else os.path.join(os.getcwd(),'weights','{}.h5'.format(iden))




if os.path.exists(WEIGHT_PATH):
  print('FOUND PRETRAINED WEIGHTS')
  LOAD_WEIGHTS=True 
else:
  print('NO PRETRAINED WEIGHTS FOUND')
  LOAD_WEIGHTS=False

 


if iden=='densenet201':
  IMG_DIM=221
  TRANSFER_LEARNING='imagenet'
if iden=='inceptionResNetv2':
  IMG_DIM=139
  TRANSFER_LEARNING='imagenet'
if iden=='inceptionv3':
  IMG_DIM=139 
  TRANSFER_LEARNING='imagenet'
if iden=='resnet101v2':
  IMG_DIM=64
  TRANSFER_LEARNING='imagenet'
if iden=='resnext101':
  IMG_DIM=64
  TRANSFER_LEARNING='imagenet'
if iden=='xception':
  IMG_DIM=71 
  TRANSFER_LEARNING='imagenet'
  
if iden=='densenet201':
  base_model_wrapper=tf.keras.applications.DenseNet201
if iden=='inceptionResNetv2':
  base_model_wrapper=tf.keras.applications.InceptionResNetV2
if iden=='inceptionv3':
  base_model_wrapper=tf.keras.applications.InceptionV3
if iden=='resnet101v2':
  base_model_wrapper=tf.keras.applications.ResNet101V2
if iden=='resnext101':
  base_model_wrapper=ResNeXt101
if iden=='xception':
  base_model_wrapper=tf.keras.applications.Xception





eval_ds = data_input_fn("Eval",BUFFER_SIZE,BATCH_SIZE,IMG_DIM,DATA_PATH=args.training_data)
train_ds = data_input_fn("Train",BUFFER_SIZE,BATCH_SIZE,IMG_DIM,DATA_PATH=args.training_data)



# # model creation

model = create_model(IMG_DIM,NB_CLASS,NB_CHANNEL)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
if LOAD_WEIGHTS:
    model.load_weights(WEIGHT_PATH)






# # Training

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,cooldown= 10,patience=5,verbose =1,min_lr=0.1e-9)
mode_autosave = tf.keras.callbacks.ModelCheckpoint(WEIGHT_PATH,save_best_only=True, verbose=0)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=8, verbose=1, mode = 'auto') 
callbacks = [mode_autosave, lr_reducer,early_stopping]




history = model.fit(train_ds,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=eval_ds,
                    validation_steps=EVAL_STEPS,
                    callbacks=callbacks)


if args.save_location != None:
    path = args.save_location
    if os.path.exists(path) and os.path.isdir(path) :
        SAVE_PATH= os.path.join(path,'new_{}.h5'.format(iden))        if os.path.isabs(path)  else        os.path.join(os.getcwd(),path.strip('./'),'new_{}.h5'.format(iden))
    elif os.path.exists(''.join(path.split('/')[:-1])):
        SAVE_PATH= path if os.path.isabs(path)  else os.path.join(os.getcwd(),path.strip('./'))
    else:
        SAVE_PATH= os.path.join(os.getcwd(),'weights','new_{}.h5'.format(iden))
        print(f'wrong link saving to {SAVE_PATH}')
else:
    SAVE_PATH= os.path.join(os.getcwd(),'weights','new_{}.h5'.format(iden))
model.save_weights(SAVE_PATH)

tp,fp,tn,fn=0,0,0,0
for x_test,y_test in tqdm(eval_ds.take(EVAL_STEPS),total=EVAL_STEPS):
  y_pred=model.predict_on_batch(x_test)
  for yt,yp in zip(y_test,y_pred):
    clp=np.argmax(yp)
    clt=np.argmax(yt)
    if clt==0:
      if clp==0:
        tn+=1
      else:
        fn+=1
    else:
      if clp==1:
        tp+=1
      else:
        fp+=1

accuracy = (tp+tn)/(tp+tn+fp+fn)
print()
print('Model:',iden)
print('================================')
print('TP: ', tp, ' TN: ', tn, ' FP: ', fp, ' FN: ', fn)
print('accuracy: ', accuracy)
    

