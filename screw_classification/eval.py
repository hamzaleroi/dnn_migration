#!/usr/bin/env python
# coding: utf-8
import efficientnet.tfkeras as efn
import warnings
import tensorflow as tf
import json
from tqdm.notebook import tqdm
import cv2
import numpy as np
import os
import argparse

class COLORS:
    red = (0, 0, 255)
    blue = (255, 0, 0)
    green = (0, 255, 0)


def detect_circles(im, **kwargs):
    x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (0, 0), fx=.3, fy=.3)
    circles = cv2.HoughCircles(x, cv2.HOUGH_GRADIENT, **kwargs)[0]
    if circles is not None:
        return (circles / .3).astype(int)
    else:
        return None


def cirlces_to_boxes(circles):
    bboxes = []
    for circle in circles:
        (x, y, r) = circle
        bbox = [int(x - r), int(y - r), int(x + r), int(y + r)]
        bboxes.append(bbox)
    return bboxes


def draw_circles(im, circles):
    out = im.copy()
    for x, y, r in circles:
        cv2.circle(out, (x, y), r, COLORS.green, 5)
    return out


def draw_preds(im, preds, circles, sz=2, thick=4, color=COLORS.green):
    im = im.copy()
    for (x, y, r), p in zip(circles, preds):
        if p != 'non_screw':
            cv2.putText(
                im,
                p,
                (x + r,
                 y + r),
                cv2.FONT_HERSHEY_COMPLEX,
                sz,
                color,
                thick,
                cv2.LINE_AA)
    return im


def cut_rois(im, circles):
    rois = []
    y_max, x_max, _ = im.shape
    for x, y, r in circles:
        up, down = max(y - r, 0), min(y + r, y_max)
        left, right = max(x - r, 0), min(x + r, x_max)
        rois.append(im[up:down, left:right].copy())
    return rois


def draw_points(im, pnts):
    out = im.copy()
    for pnt in pnts:
        cv2.circle(out, tuple(pnt), 10, COLORS.red, 30)
    return out


def draw_gt(im, pnts, lbls):
    'Draws ground-truth points'
    out = im.copy()
    if len(pnts):
        out = draw_points(out, pnts)
        # draw_preds needs circles, append radius to pnts
        r = -30 * np.ones(len(pnts), dtype=int)
        circles = np.concatenate((pnts, r.reshape(-1, 1)), axis=-1)
        out = draw_preds(out, lbls, circles, color=COLORS.red, thick=6)
    return out


def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp,
                'false_negative': fn}, []
    if len(all_gt_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp,
                'false_negative': fn}, []

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp,
                'false_negative': fn}, []
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp,
            'false_negative': fn}, ious


def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for res in image_results:
        true_positive += res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive / (true_positive + false_positive)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_positive / (true_positive + false_negative)
        except ZeroDivisionError:
            recall = 0.0
    return precision, recall

def calc_hough_res():
    with open(scenes_path, 'r') as f:
        meta = json.load(f)
    print('Found Annotations for:', len(meta), 'files')
    img_results = []
    iou_res = []
    for file_meta in tqdm(meta):
        if len(meta[file_meta]['regions']) != 0:
            # circles and boxes
            img_path = os.path.join(scenes_dir, meta[file_meta]['filename'])
            if not os.path.exists(img_path):
                continue            
            img = cv2.imread(img_path)
            circles = detect_circles(img, dp=1, **hough_params)
            if circles is not None:
                bbox_det = cirlces_to_boxes(circles)
                bbox_gt = []
                regs = meta[file_meta]['regions']
                for reg in regs:
                    x, y, w, h = reg['shape_attributes']['x'], reg['shape_attributes'][
                        'y'], reg['shape_attributes']['width'], reg['shape_attributes']['height']
                    bbox_gt.append([x, y, x + w, y + h])
                res, ious = get_single_image_results(
                    bbox_gt, bbox_det, iou_thr=0.5)

                img_results.append(res)
                if len(ious) != 0:
                    for iou in ious:
                        iou_res.append(iou)
    Precision, Recall = calc_precision_recall(img_results)
    Hough_F1Score = 2 * (Recall * Precision) / (Recall + Precision)
    print('Hough F1 Score:', Hough_F1Score)
    print('Hough Precision:', Precision)
    print('Hough Recall:', Recall)
    print('Screw Mean IoU:', np.mean(np.array(iou_res)))


def create_det_model(iden, NB_CHANNEL, WEIGHT_PATH):
    if iden == 'inceptionv3':
        base_model_wrapper = tf.keras.applications.InceptionV3
        IMG_DIM = 139
    if iden == 'xception':
        base_model_wrapper = tf.keras.applications.Xception
        IMG_DIM = 71

    base_model = base_model_wrapper(include_top=False,
                                    weights=None,
                                    input_shape=(IMG_DIM, IMG_DIM, NB_CHANNEL))
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(
        inputs=base_model.input, outputs=x, name=iden)
    model.load_weights(WEIGHT_PATH)
    return model, IMG_DIM


def create_model(IMG_DIM, NB_CHANNEL, WEIGHT_PATH, NB_CLASS):
    base_model_wrapper = efn.EfficientNetB2
    base_model = base_model_wrapper(include_top=False,
                                    weights=None,
                                    input_shape=(IMG_DIM, IMG_DIM, NB_CHANNEL))
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(NB_CLASS, activation='softmax')(x)
    model = tf.keras.models.Model(
        inputs=base_model.input, outputs=x, name=iden)
    model.load_weights(WEIGHT_PATH)
    return model
# data idens and predictions
def create_dir(_path):
  if not os.path.exists(_path):
    os.mkdir(_path)
  return _path

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




scenes_path = os.path.join(args.eval_data, 'screw.json')  if os.path.exists(args.eval_data ) else os.path.join(os.getcwd(), 'data', 'screw.json')
scenes_dir =os.path.join(args.eval_data, 'scenes')  if os.path.exists(args.eval_data ) else os.path.join(os.getcwd(), 'data', 'scenes')


hough_upper_threshold = 100  # @param {type:"slider", min:0, max:100, step:1}
hough_lower_threshold = 50  # @param {type:"slider", min:0, max:100, step:1}
hough_min_radius = 5  # @param {type:"slider", min:0, max:100, step:1}
hough_max_radius = 30  # @param {type:"slider", min:0, max:100, step:1}


hough_params = dict(minDist=100,
                    param1=hough_upper_threshold,
                    param2=hough_lower_threshold,
                    minRadius=hough_min_radius,
                    maxRadius=hough_max_radius)

calc_hough_res()



warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


iden = 'efficientnetb2'
det_iden1 = 'inceptionv3'
det_iden2 = 'xception'

integrated = [iden,det_iden1,det_iden2]

WEIGHTS_INTGRATED=[(os.path.join(args.saved_weights,'{}.h5'.format(iden)  ) \
if os.path.isabs(args.saved_weights)\
 else os.path.join(os.getwd(), args.saved_weights.strip('./'),'{}.h5'.format(iden)  ))\
 if args.saved_weights != None else os.path.join(os.getcwd(),'weights','{}.h5'.format(iden))
                  for iden in integrated]

WEIGHT_PATH, WEIGHT_PATH_DET1, WEIGHT_PATH_DET2 = WEIGHTS_INTGRATED

IMG_DIM = 256  # @param

class_names = ['ph1',
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

NB_CLASS = len(class_names)
NB_CHANNEL = 3  # @param


model = create_model(IMG_DIM, NB_CHANNEL, WEIGHT_PATH, NB_CLASS)
# det_model1,DET_DIM1=create_det_model(det_iden1,NB_CHANNEL,WEIGHT_PATH_DET1)
det_model2, DET_DIM2 = create_det_model(
    det_iden2, NB_CHANNEL, WEIGHT_PATH_DET2)

print('Classification Model:', iden)
#print('Detection Model 1:',det_iden1)
print('Detection Model 2:', det_iden2)


# ## Predictions
# It's the scene overlayed with the found circles. The found screws are
# overlayed with a shaded region

# In[ ]:



save_dir=create_dir(os.path.join(os.getcwd(),'predictions'))

with open(scenes_path, 'r') as f:
    meta = json.load(f)
# Draw
for file_meta in tqdm(meta):
    img_path = os.path.join(scenes_dir, meta[file_meta]['filename'])
    if not os.path.exists(img_path):
        continue
    else:
        print(f'processing {img_path}')
    img = cv2.imread(img_path)
    circles = detect_circles(img, dp=1, **hough_params)
    if circles is not None:
        regs = meta[file_meta]['regions']
        pnts = []
        lbls = []
        for reg in regs:
            x, y, w, h = reg['shape_attributes']['x'], reg['shape_attributes'][
                'y'], reg['shape_attributes']['width'], reg['shape_attributes']['height']
            reg_type = reg["region_attributes"]["screwtype"]
            center_x, center_y = int(x + w / 2), int(y + h / 2)
            pnts.append([center_x, center_y])
            lbls.append(reg_type)
        gtim = draw_gt(img, pnts, lbls)

        imgs = cut_rois(img, circles)
        #mgs = [cv2.cvtColor(d,cv2.COLOR_BGR2RGB) for d in imgs]
        preds = []
        pred_circles = []
        for roi, circle in zip(imgs, circles):
            '''
            roi1=cv2.resize(roi,(DET_DIM1,DET_DIM1))
            roi1=roi1.astype('float32')/255.0
            roi1=np.expand_dims(roi1,axis=0)
            det1=det_model1.predict([roi1])[0]
            '''
            roi2 = cv2.resize(roi, (DET_DIM2, DET_DIM2))
            roi2 = roi2.astype('float32') / 255.0
            roi2 = np.expand_dims(roi2, axis=0)
            det2 = det_model2.predict([roi2])[0]

            if det2[1] > 0.9:

                img_roi = cv2.resize(roi, (IMG_DIM, IMG_DIM))
                img_roi = np.expand_dims(img_roi, axis=0)
                img_roi = img_roi.astype('float32') / 255.0
                idx = np.argmax(model.predict(img_roi)[0])
                preds.append(class_names[idx])
                pred_circles.append(circle)
            else:
                preds.append('non_screw')

        pim = draw_preds(gtim, preds, circles)
        final = draw_circles(pim, pred_circles)
        cv2.imwrite(os.path.join(save_dir, meta[file_meta]['filename']), final)

    else:
        print(
            'Hough Didnot Detect Any Circle for:',
            meta[file_meta]['filename'])
