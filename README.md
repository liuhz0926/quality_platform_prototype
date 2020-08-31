# Symanto Quality Platform Prototype

This Quality Platform project is designed to unify and homogenize the evaluation of the language model. The project uses the Django framework. 

The platform contains three steps of methodology: Evaluate, Stress, and Retrain.
* Evaluate: evaluate user's language models, compare them and analyze the error
* Stress: use What-if scenarios to stress user's models and find improvements points
* Retrain: retrain user's models with the acquired knowledge and assign the proper weights

## Updates

On August 21st, 2020. The prototype contains the first step: Evaluate for both uploading predict files or using Coco Server to train

## Environment

Language: Python 3

Platform: Django 3.1

## Package install

Run the following code in command line

pip3 install -r requirements.txt

## Run server

Run the following code in command line

python3 manage.py runserver

Then click on the link generated by the code above to navigate in the server.

## Evaluate

To evaluate user's models, there are two options:
* Evaluate with a predicted file
* Evaluate with a pretrain file: train the model by using Coco Server and evaluate the result

### Evaluate with a predicted file

First, upload the truth file and predicted file to the server. Then it would evaluate the model and provide the reports. 

In the reports, the user can reupload another predicted file from a different model to compare the evaluation result of two models. 

### Evaluate with a pretrain file

First, upload a zip file with training, validation, and test dataset, and select the language model to train. The zip file would be uploaded to Coco server and start to train. After training, Coco will produce the predicted file. The backend will generate the evaluation reports based on the predicted file.

Link of Symanto Quality Platform API (Coco Server): http://symanto-pastaepizza.northeurope.cloudapp.azure.com:8000/docs

### Evaluation Reports

The reports contain four tabs:
* Overview: an evaluation table containing accuracy, precision, recall, F1-scores, etc. 
* Confusion Matrix: heap maps of the predicted labels and the truth labels in count or percentage
* Threshold Analysis: analyze the change of accuracy by increasing the threshold in 5% every time
* Error Analysis: a table containing wrong predicted data and its content
* Upload Another Predicted File (for evaluating with a predicted file mode only)

# Notes

## What is left to be developed 

* Take the server running on local machine to run on cloud.
* Stressing the model with different test set.
* Retrain a new model after modification on weights and parameters.
* User group management and Admin management. This can be done in Django.

## Other notes

* Pretrain Mode:
- Currently, since the app hasn't run on the cloud successfully, in Coco_request.py, we are using the zip file in this git (URL: https://github.com/liuhz0926/quality_platform_prototype/raw/master/uploads/evaluate/pretrain_file/archive.zip). Although we have to upload the file as well when we submit the model, the Coco API is still reading this URL from the git since Coco API cannot read a local path. Once the app is available on the server, the dataset_url needs to be updated to the new path.

- To upload the model, right now, we set in the default model requirement for Coco. Since Coco is using a small server, for now, other models may crash the API. In that situation, we might need to reload the API.
