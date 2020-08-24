# Symanto Quality Platform Prototype

This Quality Platform project is designed to unify and homogenize the evaluation of the language model. The project uses the Django framework. 

The platform contains three steps of methodology: Evaluate, Stress, and Retrain.
* Evaluate user's language models, compare them and analyze the error
* Use What-if scenarios to stress user's models and find improvements points
* Retrain user's models with the acquired knowledge and assign the proper weights

## Updates

On August 21st, 2020. The prototype contains the first step, Evaluate for both uploading predict files or using Coco Server to train

## Environment

Language: Python 3

Platform: Django 3.1

## Evaluate

To evaluate user's models, there are two options:
* Evaluate with a predicted file
* Evaluate with a pretrain file: train the model by using Coco Server and evaluate the result

### Evaluate with a predicted file

First, upload the truth file and predicted file to the server. Then it would evaluate the model and provide the reports. 

In the reports, the user can reupload another predicted file from a different model to compare the evaluation result of two models. 

### Evaluate with a pretrain file

First, upload a zip file with training, validation, and test dataset, and select the language model to train. The zip file would be uploaded to Coco server and start to train. After training, Coco will produce the predicted file. The backend will generate the evaluation reports based on the predicted file.

### Evaluation Reports

The reports contain four tabs:
* Overview: an evaluation table containing accuracy, precision, recall, F1-scores, etc. 
* Confusion Matrix: heap maps of the predicted labels and the truth labels in count or percentage
* Threshold Analysis: analyze the change of accuracy by increasing the threshold in 5% every time
* Error Analysis: a table containing wrong predicted data and its content
* Upload Another Predicted File (for evaluating with a predicted file mode only)
