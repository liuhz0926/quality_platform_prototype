from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import time
from .forms import EvalFileForm, EvalAddFileForm, EvalPretrainForm
from .models import EvalPredFile, EvalAddFile, EvalPretrainFile
from .backend.Main import Eval_Report
from .backend.evaluate.Coco_Request import Coco_request


EVAL_REPORT = Eval_Report()

def load_backend(prediction=False, pretrain=False, addition=False):
    home_address = '~/quality_platform_prototype/'

    # get the last uploaded files
    if prediction:
        uploaded_files = EvalPredFile.objects.last()
        truth_url = uploaded_files.truth_file.url
        prediction_url = uploaded_files.prediction_file.url
        truth_file = home_address + truth_url
        prediction_file = home_address + prediction_url
        pretrain_labels = None

    # UNDER CONSTRUCTION
    if pretrain:
        pretrain_labels, prediction_file = load_coco()
        #print(prediction_file)
        time.sleep(2)
        truth_file = None


    #print(truth_file, prediction_file)

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
    # UNDER CONSTRUCTION
    # #test.post_train()
    #     #test.get_status()
    #     #test.post_predict()
    #
    #     #pretrain_form = EvalPretrainFile.objects.last()
    #     #print(pretrain_form)
    #     #print(pretrain_form.tokenization)
    #     #print(pretrain_form.pretrain_file.url)
    #     #print(pretrain_form.description)
    #     #print(pretrain_form.n_classes)
    #     #print(pretrain_form.pretrain_file.name)
    coco_model = Coco_request()
    coco_model.post_train()
    coco_model.check_status()
    coco_model.post_predict()
    return coco_model.train_labels, coco_model.predict_file


def set_report_title():
    '''

    :return: set up the context dictionary for the report titile
    '''
    if EVAL_REPORT.predict:
        return {'title': 'Report with a Prediction File'}
    if EVAL_REPORT.pretrain:
        return {'title': 'Report after training from Coco API'}



def home(request):
    '''
    :param request:
    :return: Home page
    '''
    context = {}
    return render(request, 'quality_platform/platform_home.html', context)


def evaluate(request):
    '''
    :param request:
    :return: evaluate home page
    '''
    return render(request, 'quality_platform/evaluate_base.html')


def eval_upload_prediction(request):
    '''
        After clicking with a prediction file
        upload truth file and prediction file
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
            load_backend(prediction=True)
            return redirect('platform-evaluate-report')

    else: # if request is get
        form = EvalFileForm()
        context['form'] = form

    return render(request, 'quality_platform/eval_upload_prediction.html', context)


def eval_report_overview(request):
    '''
        Find the two uploaded files and call the Overview Class from the backend
        Calculate the overview and produce the confusion matrix png
    :param request:
    :return: render request to the overview page
    '''
    context = set_report_title()

    # This is temporary for unfinished pretrain part (Need to delete after finish pretrain)
    if EVAL_REPORT.evaluate_table is None:
        return render(request, 'quality_platform/eval_report_overview.html', context)

    # make an evaluation table
    context['evaluation'] = EVAL_REPORT.evaluate_table
    context['total_instance'] = EVAL_REPORT.total_instance
    context['instance_per_class'] = EVAL_REPORT.instance_class

    if EVAL_REPORT.add_total_instance != None:
        context['add_evaluation'] = EVAL_REPORT.add_evaluate_table
        context['add_total_instance'] = EVAL_REPORT.add_total_instance
        context['add_instance_per_class'] = EVAL_REPORT.add_instance_class


    return render(request, 'quality_platform/eval_report_overview.html', context)


def eval_report_confusion(request):
    '''
        Since the overview page would go first,
        Overview page would create the confusion matrix png in the static folder
        Call it in the html directly
    :param request:
    :return: render request to the confusion matrix page
    '''
    context = set_report_title()

    # This is temporary for unfinished pretrain part (Need to delete after finish pretrain)
    if EVAL_REPORT.evaluate_table is None:
        return render(request, 'quality_platform/eval_report_overview.html', context)

    context['confusion_labels'] = EVAL_REPORT.confusion_labels
    context['confusion_data'] = EVAL_REPORT.confusion_data

    if EVAL_REPORT.add_confusion_labels:
        context['add_confusion_labels'] = EVAL_REPORT.add_confusion_labels
        context['add_confusion_data'] = EVAL_REPORT.add_confusion_data

    return render(request, 'quality_platform/eval_report_confusion.html', context)


def eval_report_confusion_proportion(request):
    '''
        Since the overview page would go first,
        Overview page would create the confusion matrix proportion png in the static folder
        Call it in the html directly
    :param request:
    :return: render request to the confusion matrix page
    '''
    context = set_report_title()

    # This is temporary for unfinished pretrain part (Need to delete after finish pretrain)
    if EVAL_REPORT.evaluate_table is None:
        return render(request, 'quality_platform/eval_report_overview.html', context)

    context['normal_labels'] = EVAL_REPORT.normal_labels
    context['normal_data'] = EVAL_REPORT.normal_data

    if EVAL_REPORT.add_confusion_labels:
        context['add_normal_labels'] = EVAL_REPORT.add_normal_labels
        context['add_normal_data'] = EVAL_REPORT.add_normal_data
    return render(request, 'quality_platform/eval_report_confusion_proportion.html', context)


def eval_report_threshold(request):
    '''

    :param request:
    :return:
    '''
    context = set_report_title()

    # This is temporary for unfinished pretrain part (Need to delete after finish pretrain)
    if EVAL_REPORT.evaluate_table is None:
        return render(request, 'quality_platform/eval_report_overview.html', context)

    context['threshold'] = EVAL_REPORT.threshold
    context['threshold_list'] = EVAL_REPORT.threshold_list
    context['threshold_accuracy'] = EVAL_REPORT.threshold_accuracy

    if EVAL_REPORT.add_threshold_accuracy:
        context['add_threshold_accuracy'] = EVAL_REPORT.add_threshold_accuracy

    return render(request, 'quality_platform/eval_report_threshold.html', context)


def eval_report_error(request):
    '''

    :param request:
    :return:
    '''
    context = set_report_title()

    # This is temporary for unfinished pretrain part (Need to delete after finish pretrain)
    if EVAL_REPORT.evaluate_table is None:
        return render(request, 'quality_platform/eval_report_overview.html', context)

    context['error'] = EVAL_REPORT.error
    if EVAL_REPORT.add_threshold_accuracy:
        context['addition'] = 1
    return render(request, 'quality_platform/eval_report_error.html', context)


def eval_report_upload(request):
    '''

    :param request:
    :return:
    '''
    context = set_report_title()

    # This is temporary for unfinished pretrain part (Need to delete after finish pretrain)
    if EVAL_REPORT.evaluate_table is None:
        return render(request, 'quality_platform/eval_report_overview.html', context)

    if request.method == 'POST':
        form = EvalAddFileForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            # Evaluate the model
            load_backend(prediction=True, addition=True)
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
            load_backend(pretrain=True)
            return redirect('platform-evaluate-report')
        pass
    else: # if request is get
        form = EvalPretrainForm()
        context['form'] = form

    return render(request, 'quality_platform/eval_upload_pretrain.html', context)


