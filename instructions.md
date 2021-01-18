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
# Getting the data:
The scripts need data to function properly. In order to download the dataset for each dnn execute the script download_from_drive.sh this way:  
```bash
sh download_from_drive.sh ids.txt
```

