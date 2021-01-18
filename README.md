# dnn_migration
This is where goes thee refactored DNN models for screws and wires
# Explaining the folders:

Each one of the folders contain the following files:

1.  `data_join.py`
 * Organises image files into one folder

2.  `data_store.py`
 *  Converts a folder of images into a h5 format

3.  `data.py`
 *  Converts the .h5 files into .tfrecord in order to allow the training on these files.

4.  `train.py`
 *  Allows the training on the .tfrecord files
 *  Note: for classification, a GPU with fairly good capacity or a TPU is needed for training.

5.  `infer.py`
 *  Allows performing inference on an image
 *  It is stored in /tmp by default.

6.  `eval.py`
 *  Evaluates the performance of the model.

For more info on these files use the `--help` option to display the command tool syntax.  


# First-time  folder preparation:
## Getting the data:
The scripts need data to function properly. In order to download the dataset for each dnn execute the script download_from_drive.sh this way:  
```bash
sh download_from_drive.sh ids.txt
```

## Building repository with custom dataset:  
Before downloading the data, make sure the structure of teh folders are respected:  


-> screw_classification   
---> ScrewCTF  

-> screw_detection  
----> ScrewCTF  

-> wire_detection   
----> WireDTF   

To download the new custom dataset:

1. Place  the ids of the tar files in the file ids.txt in the following format:  
`<id>,<name>`  
Note the name should match the corresponding folder in this current folder. Also, there should be one new line per id in order to be considered. In other words, every new line ''\n' correspond to one file to be downloaded.

2. run the script `download_from_drive` giving it `idx.txt` as first parameter
3. the script should un-compress the data in the corresponding folders


# How to add data to the model:  
1. Place data in the corresponding folder.
2. Run the script `data_join.py` of the corresponding DCNN. This will pick the valid images and put them in a special folder.
3. Run the script `data_score.py` of thecorresponding DCNN. The script is going to take the last folder and produce files with the .h5 extension


