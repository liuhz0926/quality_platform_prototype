import json

def config_to_python(description="None",dataset="DE-sentiment-mixed",data_folder="/data/research/users/abasile/pseudo-home/coco-data", tokenization="transformer",architecture="transformer",pretrained_model="bert-base-cased",finetune=True,max_length=100,epochs=15,n_classes=2,labels=[1,0]):
    my_json = {

    "description": description,

    "dataset": dataset,

    "dataset_reader": "tfdataset",

    "data_folder": data_folder,

    "tokenization": [tokenization

    ],

    "architecture": architecture,

    "batch_size": 50,

    "pretrained_model": pretrained_model,

    "finetune": finetune,

    "lr": 3e-5,

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
    with open('config.json','w') as file:
        json.dump(my_json,file,ensure_ascii=False)
    return None
config_to_python()