#!/usr/bin/env python
import imageio
import os
import numpy as np
import cv2
from PIL import Image as imgop
from glob import glob
from sklearn.metrics import jaccard_similarity_score
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse



def get_img(_path):
    data = imgop.open(_path)
    data = data.resize((IMG_DIM, IMG_DIM))
    data = np.array(data)
    data = data.astype('float32') / 255.0
    data = np.expand_dims(data, axis=0)
    return data


def get_gt(_path):
    # test folder mask path
    _mpath = str(_path).replace("images", "masks")
    # ground truth
    gt = cv2.imread(_mpath, 0)
    # resize
    gt = cv2.resize(gt, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gt, (5, 5), 0)
    _, gt = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gt


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


def get_score(pred, gt):
    (ssim_score, _) = compare_ssim(gt, pred, full=True)
    iou = jaccard_similarity_score(gt.flatten(), pred.flatten())
    return ssim_score, iou


def score_summary(arr, model_name, score_iden):
    print(model_name, ':', score_iden)
    print('max:', np.amax(arr))
    print('mean:', np.mean(arr))
    print('min:', np.amin(arr))

def data_input_fn(mode,BUFFER_SIZE,BATCH_SIZE,IMG_DIM,data_img_dim=64,DATA_PATH='ScrewDTF'): 
def data_input_fn(mode,BUFFER_SIZE,BATCH_SIZE,img_dim,DATA_PATH='ScrewDTF'): 
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'target' : tf.io.FixedLenFeature([],tf.string)
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=3)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(img_dim,img_dim,3))
        
        
        target_raw=parsed_example['target']
        target=tf.image.decode_png(target_raw,channels=1)
        target=tf.cast(target,tf.float32)/255.0
        target=tf.reshape(target,(img_dim,img_dim,1))
        
        return image,target
    files_pattern=  DATA_PATH if os.path.isabs(DATA_PATH)\
      else os.path.join(os.getcwd(),DATA_PATH.strip('./'),mode,'*.tfrecord')
    file_paths = tf.io.gfile.glob(files_pattern)
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset   


def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(
            activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3

    def wrapper(input_tensor):

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(
                axis=bn_axis, name=bn_name)(x)

        if activation:
            x = tf.keras.layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper


def Conv3x3BnReLU(filters, use_batchnorm, name=None):

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name)(input_tensor)
    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3

    def layer(input_tensor, skip=None):

        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(
                axis=bn_axis, name=bn_name)(x)

        x = tf.keras.layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = tf.keras.layers.Concatenate(
                axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

def effcientb7_unet(input_shape):
    encoder_features = ('block6a_expand_activation',
                        'block4a_expand_activation',
                        'block3a_expand_activation',
                        'block2a_expand_activation')

    backbone = tf.keras.applications.EfficientNetB7(include_top=False,
                                                    input_shape=input_shape,
                                                    weights='imagenet')
    outs = []
    input_ = backbone.input
    x = backbone.output
    # extract skip connections
    skips = ([backbone.get_layer(name=i).output for i in encoder_features])
    # build decoders
    decoder_filters = (256, 128, 64, 32, 16)
    n_upsample_blocks = len(decoder_filters)
    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = DecoderTransposeX2Block(
            decoder_filters[i],
            stage=i,
            use_batchnorm=True)(
            x,
            skip)

    mask = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv_mask',
    )(x)
    mask = tf.keras.layers.Activation('sigmoid', name='mask')(mask)
    model = tf.keras.models.Model(
        input_, [mask, backbone.output], name='efficient_unet')
    return model
def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def create_dir(base_dir, ext_name):
    '''
        creates a new dir with ext_name in base_dir and returns the path
    '''
    new_dir = os.path.join(base_dir, ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training_data',type = str, default='WireDTF', help = 'The model location')
    
    parser.add_argument('--save_location', type = str, help = 'Where to save the trained model')
    
    parser.add_argument('--saved_weights', type = str,  help = 'Load the pre-trained weights')
     
    parser.add_argument('--batch_size', type = int, default=128, help = 'batch_size')
     


    in_args = parser.parse_args()

    return in_args





if __name__ == '__main__':
    args = get_input_args()
    model_name='efficientnetb7'
    iden='model'
    IMG_DIM = 256  # @param
    NB_CHANNEL = 3  # @param
    BATCH_SIZE= args.batch_size # @param
    BUFFER_SIZE = 2048  # @param
    TRAIN_DATA = 10240  # @param
    EVAL_DATA = 1024  # @param
    EPOCHS = 100  # @param
    TOTAL_DATA = TRAIN_DATA + EVAL_DATA
    STEPS_PER_EPOCH = TOTAL_DATA // BATCH_SIZE
    EVAL_STEPS = EVAL_DATA // BATCH_SIZE


    WEIGHT_PATH= args.saved_weights if args.saved_weights != None else os.path.join(os.getcwd(),'weights','{}.h5'.format(iden))
    if os.path.exists(WEIGHT_PATH):
        print('FOUND PRETRAINED WEIGHTS')
        LOAD_WEIGHTS = False  # @param
    else:
        print('NO PRETRAINED WEIGHTS FOUND')
        LOAD_WEIGHTS = False


    eval_ds = data_input_fn("Eval", BUFFER_SIZE, BATCH_SIZE, IMG_DIM,DATA_PATH=args.training_data)
    train_ds = data_input_fn("Train", BUFFER_SIZE, BATCH_SIZE, IMG_DIM,DATA_PATH=args.training_data)
    for x, y in eval_ds.take(1):
        print(x.shape)
        print(y.shape)



    model = effcientb7_unet((IMG_DIM, IMG_DIM, NB_CHANNEL))
    model.compile(optimizer="Adam",
                    loss=tf.keras.losses.mean_squared_error,
                    metrics=[ssim])
    if LOAD_WEIGHTS:
        model.load_weights(WEIGHT_PATH)

    # reduces learning rate on plateau
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                    cooldown=10,
                                                    patience=10,
                                                    verbose=1,
                                                    min_lr=0.1e-5)

    mode_autosave = tf.keras.callbacks.ModelCheckpoint(WEIGHT_PATH,
                                                    monitor='val_ssim',
                                                    mode='max',
                                                    save_best_only=True,
                                                    verbose=1,
                                                    period=10)

    # stop learining as metric on validatopn stop increasing
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=15,
                                                    verbose=1,
                                                    mode='auto')

    callbacks = [mode_autosave, lr_reducer]  # ,early_stopping ]


    history = model.fit(train_ds,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=eval_ds,
                        validation_steps=EVAL_STEPS,
                        callbacks=callbacks)

    # save model
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

    img_dir = os.path.join(os.getcwd(), 'source', 'base', 'test', 'images')
    tgt_dir = os.path.join(os.getcwd(), 'source', 'base', 'test', 'masks')



    pred_path = create_dir(os.getcwd(), 'test')

    img_paths = glob(os.path.join(img_dir, '*.*'))
    SSIM = []
    IOU = []
    # inference model
    mod = effcientb7_unet((IMG_DIM, IMG_DIM, NB_CHANNEL))
    mod.load_weights(WEIGHT_PATH)
    print('Loaded inference weights')
    for _path in img_paths:
        # ground truth
        gt = get_gt(_path)
        # image
        img = get_img(_path)
        # prediction
        pred = get_pred(mod, img, _path)
        # overlay
        overlay = get_overlay(pred, img)
        # scores
        ssim_score, iou = get_score(pred, gt)
        SSIM.append(ssim_score)
        IOU.append(iou)
        imageio.imsave(os.path.join(pred_path, os.path.basename(_path)), overlay)

    print(
        'The saved predictions can be found at:{} folder'.format(
            os.path.join(
                os.getcwd(),
                'test')))

    score_summary(np.array(SSIM), model_name, 'ssim')
    score_summary(np.array(IOU), model_name, 'IoU/F1')
