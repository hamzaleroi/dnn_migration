
# Running classification:

## Clone the repo:  
```bash
git clone https://github.com/hamzaleroi/dnn_migration
```


## Download the data
```bash
cd dnn_migration && sh download_from_drive.sh ids.txt
```


## Installing dependancies:
```bash
pip install -r /content/dnn_migration/requirements.txt
```


## Testing the script data.py

This script converts .h5 into tfrecord to be trained on. You can find them on your drive under SCREW_CLASSIFICATION. provide that link to the data.py script with the `--data_location` command.  
The script will look for files among starting with one from :  ['ph1', 'slotted6.5', 'torx7', 'allen2.75', 'ph2', 'allen4', 'torx8', 'slotted4.5', 'torx9', 'torx6', 'slotted10', 'allen2.5'] and ending with `.h5`. This is going to be stored in `.tfrecord` directly.




```bash
python /content/dnn_migration/screw_classification/data.py --data_location /content/drive/MyDrive/SCREW_CLASSIFICATION/data
```

## Testing train.py:  

Once you have the data you can start training. Note that the model selected here will require a more powerful GPU and the one on colab will not work. Using a TPU however is still possible.

```bash

```
```bash
python /content/dnn_migration/screw_classification/train.py --training_data /content/dnn_migration/screw_classification/ScrewCTF
```

##  Testing infer.py :  
Inference will store the resulting image in /tmp. The file name will be proceded by 'inference_'

```bash
wget https://cdn.techgyd.com/Hard-Drive.png
```  

```bash
python /content/dnn_migration/screw_classification/infer.py ./Hard-Drive.png \
     --classification_model_location  /content/dnn_migration/screw_classification/weights\
     --detection_model_location  /content/dnn_migration/screw_classification/weights
```

```bash
cp  /tmp/inference_Hard-Drive.png .
```

## Evaluating the model:


```bash
python /content/dnn_migration/screw_classification/eval.py \
       --eval_data /content/drive/MyDrive/SCREW_CLASSIFICATION/data\
       --saved_weights /content/dnn_migration/screw_classification/weights
       
```
