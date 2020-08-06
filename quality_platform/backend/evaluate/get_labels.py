import pandas as pd
import zipfile
import os

def unzip(zip_directory, unzip_directory):
    with zipfile.ZipFile(os.path.expanduser(zip_directory), 'r') as zip_ref:
        zip_ref.extractall(unzip_directory)

def get_labels(filename):
    file = pd.read_csv(filename,sep='\t',header=None)
    return file.iloc[:,1].value_counts().keys().tolist()
