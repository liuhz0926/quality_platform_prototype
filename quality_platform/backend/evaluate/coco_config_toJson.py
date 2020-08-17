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
    # with open('config.json','w') as file:
        # json.dump(my_json,file,ensure_ascii=False)
    return json.dumps(my_json)

# print(config_to_python())