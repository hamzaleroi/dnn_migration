# dnn_migration
This is where goes thee refactored DNN models for screws and wires
## Steps for get everything up and running:
1. Place  the ids of the tar files in the file ids.txt in the following format:  
`<id>,<name>`  
Note the name should match the corresponding folder in this current folder.

2. run the script `download_from_drive` giving it `idx.txt` as first parameter
3. the script should uncompress the data in the corresponding folders

# Explaining the folders:
## SCREW CLASSIFICATION
### infer.py:
usage: infer.py [-h]  
                [--classification_model_location   CLASSIFICATION_MODEL_LOCATION]  
                [--detection_model_location   DETECTION_MODEL_LOCATION]  
                [--infer_folder INFER_FOLDER]  
                image_path  

positional arguments:  
  image_path            Image to apply inference on  

optional arguments:  
  -h, --help            show this help message and exit  
  --classification_model_location CLASSIFICATION_MODEL_LOCATION  
                        The classification model location  
  --detection_model_location DETECTION_MODEL_LOCATION
                        The detection model location
  --infer_folder INFER_FOLDER
                        Apply inference on a complete folder``


### train.py

usage: train.py [-h] [--training_data TRAINING_DATA]
                [--save_location SAVE_LOCATION]
                [--saved_weights SAVED_WEIGHTS] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --training_data TRAINING_DATA
                        The model location
  --save_location SAVE_LOCATION
                        Where to save the trained model
  --saved_weights SAVED_WEIGHTS
                        Load the pre-trained weights
  --batch_size BATCH_SIZE
                        batch_size