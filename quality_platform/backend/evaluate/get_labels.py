import pandas as pd
import zipfile
import tarfile
import os

def unzip(zip_directory, unzip_directory):
    '''
    Extract the zip file into another directory
    :param zip_directory: the path of zip file
    :param unzip_directory: the path of the folder that intend to unzip into
    :return:
    '''

    zip_path = os.path.expanduser(zip_directory)
    unzip_path = os.path.expanduser(unzip_directory)

    if zip_path.endswith("tar.gz"):
        tar = tarfile.open(zip_path, "r:gz")
        tar.extractall(path=unzip_path)
        tar.close()

    elif zip_path.endswith("tar"):
        tar = tarfile.open(zip_path, "r:")
        tar.extractall(path=unzip_path)
        tar.close()

    elif zip_path.endswith("zip"):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=unzip_path)

def get_labels(filename):
    '''
    read the label from the training set
    :param filename: the path of training set
    :return: list of labels
    '''
    file = pd.read_csv(filename,sep='\t',header=None)
    return file.iloc[:,1].value_counts().keys().tolist()
