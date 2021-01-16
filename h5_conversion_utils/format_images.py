import shutil
import os
import argparse
import pandas as pd

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
    
    parser.add_argument('--image_folder', type = str,default=os.path.join(os.getcwd(),'images'),
     help = 'The folder containing all the images')
    
    parser.add_argument('--dest_folder', type = str, default=os.path.join(os.getcwd()),help = 'Where to sort the images according to their class')

    parser.add_argument('--json_file',   type = str, default=os.path.join(os.getcwd(),'screw.json'),help = 'Where the json file is')

    in_args = parser.parse_args()

    return in_args

args = get_input_args()

base_dir = args.image_folder
save_dir = args.dest_folder

if __name__ == "__main__":
    df = pd.read_json(args.json_file).T
    non_screws = df[df['regions'].apply(lambda x: len(x) == 0)].index
    non_screws = [''.join(index.split('.')[:-1]) + '.' +  ('png' if index.split('.')[-1].startswith('png') else index.split('.')[-1])  for index in non_screws]
    screws = df[~df['regions'].apply(lambda x: len(x) == 0)].index
    screws = [''.join(index.split('.')[:-1]) + '.' +  ('png' if index.split('.')[-1].startswith('png') else index.split('.')[-1])  for index in screws]

    save_dir_screws = create_dir(os.path.join(save_dir,'screws'))
    for screw in screws:
        shutil.copy(os.path.join(base_dir,screw),os.path.join(save_dir,save_dir_screws))

    save_dir_non_screws = create_dir(os.path.join(save_dir,'non_screw'))
    for screw in non_screws:
        shutil.copy(os.path.join(base_dir,screw),os.path.join(save_dir,save_dir_non_screws))