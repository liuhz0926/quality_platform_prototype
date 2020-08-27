import json

def config_to_python(description="None",
                     dataset="DE-sentiment-mixed",
                     data_folder="/data/research/users/abasile/pseudo-home/coco-data",
                     tokenization="char",
                     architecture="cnn_char",
                     embedding_size = 25,
                     pretrained_model="bert-base-cased",
                     finetune=True,
                     max_features = 100,
                     max_length=10,
                     epochs=15,
                     n_classes=2,
                     labels=[1,0]):
    '''
    Make the data model for Coco Server into json format
    Read the data from front end

    :param description: str
    :param dataset: str, dataset name
    :param data_folder: str
    :param tokenization: str
    :param architecture: str
    :param embedding_size: int
    :param pretrained_model: str
    :param finetune: boolean
    :param max_features:int
    :param max_length:int
    :param epochs: int
    :param n_classes: int
    :param labels: list of labels
    :return:
    '''



    my_json = {

    "description": description,

    "dataset": dataset,

    "dataset_reader": "tfdataset",

    "data_folder": data_folder,

    "tokenization": [tokenization

    ],

    "architecture": architecture,

    "batch_size": 8,

    # Only need when the model is CNN
    "embedding_size": embedding_size,

    # Only need when the model select BERT
    # "pretrained_model": pretrained_model,

    "finetune": finetune,

    "lr": 3e-05,

    "max_features": max_features,

    "max_length": max_length,

    "epochs": epochs,

    "n_classes": n_classes,

    "metrics": [

        "acc"

    ],

    "model_dir": "/data/research/users/abasile/devops/symanto-coco/models_output/de-sentiment-bert-native",

    "labels": labels,

    "GPUs": "0"

    }

    return json.dumps(my_json)

