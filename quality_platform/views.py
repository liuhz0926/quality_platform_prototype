from django.shortcuts import render, redirect
import time
from .forms import EvalFileForm, EvalAddFileForm, EvalPretrainForm
from .models import EvalPredFile, EvalAddFile, EvalPretrainFile
from .backend.Main import Eval_Report
from .backend.evaluate.Coco_Request import Coco_request
import os

EVAL_REPORT = Eval_Report()

def load_evaluate(prediction=False, pretrain=False, addition=False):
    '''
    Loading the Evaluation Reports
    First check the predicted mode or pretrain mode, contains addition predicted file or not
    Then generate the EVAL_REPORT objects and load reports from Main in the backend folder

    :param prediction: boolean
    :param pretrain: boolean
    :param addition: boolean
    :return:
    '''
    home_address = '~/quality_platform_prototype/'

    # get the last uploaded files
    if prediction:
        uploaded_files = EvalPredFile.objects.last()
        truth_url = uploaded_files.truth_file.url
        prediction_url = uploaded_files.prediction_file.url
        truth_file = os.path.expanduser(home_address + truth_url)
        prediction_file = os.path.expanduser(home_address + prediction_url)
        pretrain_labels = None

    if pretrain:
        # because after the prediction file has been produced in Coco
        # We have to guarantee that the url is valid, so wait for 1 second
        pretrain_labels, prediction_file = load_coco()
        time.sleep(1)
        truth_file = None


    if not addition:
        # init Eval_Report object
        global EVAL_REPORT
        EVAL_REPORT = Eval_Report()

        if prediction:
            EVAL_REPORT.predict = True
        if pretrain:
            EVAL_REPORT.pretrain = True


    if addition:
        addition_files = EvalAddFile.objects.last()
        add_pred_file = home_address + addition_files.addition_pred_file.url
        # set the new report
        EVAL_REPORT.load_report(truth_file, prediction_file, add_pred_file, labels = pretrain_labels)

    else:
        EVAL_REPORT.load_report(truth_file, prediction_file, labels = pretrain_labels)

    return

def load_coco():
    '''
        After uploading/submitting the pretrain files and models for Coco
        First do a Post/Train request to upload the file to Coco and start Training
        Check the Status every two seconds
        After training is finished, get the prediction
    :return: coco_model.train_labels: list of classification labels for loading reports
             coco_model.predict_file: prediction url
    '''

    coco_model = Coco_request()
    coco_model.post_train()
    coco_model.check_status()
    coco_model.post_predict()
    return coco_model.train_labels, coco_model.predict_file


def set_report_title():
    '''
    set up the context dictionary for the report titile
    :return: dictionary for context
    '''
    if EVAL_REPORT.predict:
        return {'title': 'Report with a Prediction File'}
    if EVAL_REPORT.pretrain:
        return {'title': 'Report after training from Coco API'}



def home(request):
    '''
    Load Home Page
    :param request:
    :return: Home page
    '''
    context = {}
    return render(request, 'quality_platform/platform_home.html', context)


def evaluate(request):
    '''
    Load Evaluate Home page
    :param request:
    :return: evaluate home page
    '''
    return render(request, 'quality_platform/evaluate_base.html')


def eval_upload_prediction(request):
    '''
        After clicking with a prediction file
        upload truth file and prediction file
        Loading the Evaluation Report model
        redirect to the report page
    :param request:
    :return:
    '''
    context = {'title': 'with a Prediction File'}

    if request.method == 'POST':
        form = EvalFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Evaluate the model
            load_evaluate(prediction=True)
            return redirect('platform-evaluate-report')

    else: # if request is get
        form = EvalFileForm()
        context['form'] = form

    return render(request, 'quality_platform/eval_upload_prediction.html', context)


def eval_report_overview(request):
    '''
        Generate the classification reports by getting the data from the EVAL_REPORT object
        predict use to tell the front end which type of table needs to generate
        if there is additional predicted file, it would also tell the front end to make the second table
    :param request:
    :return: render request to the overview page
    '''
    context = set_report_title()

    # make an evaluation table
    context['evaluation'] = EVAL_REPORT.evaluate_table
    context['total_instance'] = EVAL_REPORT.total_instance
    context['instance_per_class'] = EVAL_REPORT.instance_class
    context['predict'] = EVAL_REPORT.predict


    if EVAL_REPORT.add_total_instance != None:
        context['add_evaluation'] = EVAL_REPORT.add_evaluate_table
        context['add_total_instance'] = EVAL_REPORT.add_total_instance
        context['add_instance_per_class'] = EVAL_REPORT.add_instance_class


    return render(request, 'quality_platform/eval_report_overview.html', context)


def eval_report_confusion(request):
    '''
        Generate the confusion matrix in count by getting the labels and data
        predict use to tell the front end which type of table needs to generate
        if there is additional predicted file, it would also tell the front end to make the second chart
    :param request:
    :return: render request to the confusion matrix page
    '''
    context = set_report_title()

    context['confusion_labels'] = EVAL_REPORT.confusion_labels
    context['confusion_data'] = EVAL_REPORT.confusion_data
    context['predict'] = EVAL_REPORT.predict

    if EVAL_REPORT.add_confusion_labels:
        context['add_confusion_labels'] = EVAL_REPORT.add_confusion_labels
        context['add_confusion_data'] = EVAL_REPORT.add_confusion_data

    return render(request, 'quality_platform/eval_report_confusion.html', context)


def eval_report_confusion_proportion(request):
    '''
        Generate the confusion matrix in percentage by getting the labels and data
        predict use to tell the front end which type of table needs to generate
        if there is additional predicted file, it would also tell the front end to make the second chart
    :param request:
    :return: render request to the confusion matrix page
    '''
    context = set_report_title()

    context['normal_labels'] = EVAL_REPORT.normal_labels
    context['normal_data'] = EVAL_REPORT.normal_data
    context['predict'] = EVAL_REPORT.predict

    if EVAL_REPORT.add_confusion_labels:
        context['add_normal_labels'] = EVAL_REPORT.add_normal_labels
        context['add_normal_data'] = EVAL_REPORT.add_normal_data
    return render(request, 'quality_platform/eval_report_confusion_proportion.html', context)


def eval_report_threshold(request):
    '''
        Generate the threshold analysis and return the threshold table, threshold_list and accuracy_list for chart
        predict use to tell the front end which type of table needs to generate
        if there is additional predicted file, it would also tell the front end to make the second table
    :param request:
    :return: render request to the threshold analysis page
    '''
    context = set_report_title()

    context['threshold'] = EVAL_REPORT.threshold
    context['threshold_list'] = EVAL_REPORT.threshold_list
    context['threshold_accuracy'] = EVAL_REPORT.threshold_accuracy
    context['predict'] = EVAL_REPORT.predict

    if EVAL_REPORT.add_threshold_accuracy:
        context['add_threshold_accuracy'] = EVAL_REPORT.add_threshold_accuracy

    return render(request, 'quality_platform/eval_report_threshold.html', context)


def eval_report_error(request):
    '''
        Generate the error analysis and return the error table
        predict use to tell the front end which type of table needs to generate
        if there is additional predicted file, it would also tell the front end to make two more column
    :param request:
    :return: render request to the error analysis page
    '''
    context = set_report_title()

    context['error'] = EVAL_REPORT.error
    context['predict'] = EVAL_REPORT.predict
    if EVAL_REPORT.add_threshold_accuracy:
        context['addition'] = 1
    return render(request, 'quality_platform/eval_report_error.html', context)


def eval_report_upload(request):
    '''
        Upload another predicted file to the backend
        It would load the backend again and use the previous truth file and predicted file
    :param request:
    :return:
    '''
    context = set_report_title()

    if request.method == 'POST':
        form = EvalAddFileForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            # Evaluate the model
            load_evaluate(prediction=True, addition=True)
            return redirect('platform-evaluate-report')

    else: # if request is get
        form = EvalAddFileForm()
        context['form'] = form

    return render(request, 'quality_platform/eval_report_upload_new.html', context)


def eval_upload_pretrain(request):
    '''
        After clicking with a pretrain model
        upload a zip file for coco and file in the table for coco
        redirect to the report page
    :param request:
    :return:
    '''
    context = {'title': 'with a Pretrain Model'}

    if request.method == 'POST':
        form = EvalPretrainForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            load_evaluate(pretrain=True)
            return redirect('platform-evaluate-report')
        pass
    else: # if request is get
        form = EvalPretrainForm()
        context['form'] = form

    return render(request, 'quality_platform/eval_upload_pretrain.html', context)


