from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from .forms import EvalFileForm
from .models import EvalPredFile
from .backend.Main import Eval_Report


EVAL_REPORT = Eval_Report()


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
            load_backend()
            return redirect('platform-evaluate-report-prediction')

    else: # if request is get
        form = EvalFileForm()
        context['form'] = form

    return render(request, 'quality_platform/eval_upload_prediction.html', context)


def load_backend():
    home_address = '/Users/liuhz0926/Django_project/GitHub/quality_platform_prototype/'

    # get the last uploaded files
    uploaded_files = EvalPredFile.objects.last()
    truth_file = home_address + uploaded_files.truth_file.url
    prediction_file = home_address + uploaded_files.prediction_file.url

    # set the report
    global EVAL_REPORT
    EVAL_REPORT.load_report(truth_file, prediction_file)

    return


def eval_report_prediction(request):
    '''
        Find the two uploaded files and call the Overview Class from the backend
        Calculate the overview and produce the confusion matrix png
    :param request:
    :return: render request to the overview page
    '''
    context = {'title': 'Report with a Prediction File'}

    # make an evaluation table
    context['evaluation'] = EVAL_REPORT.evaluate_table
    context['total_instance'] = EVAL_REPORT.total_instance
    context['instance_per_class'] = EVAL_REPORT.instance_class

    return render(request, 'quality_platform/eval_report_prediction.html', context)


def eval_pred_report_confusion(request):
    '''
        Since the overview page would go first,
        Overview page would create the confusion matrix png in the static folder
        Call it in the html directly
    :param request:
    :return: render request to the confusion matrix page
    '''
    context = {'title': 'Report with a Prediction File'}
    return render(request, 'quality_platform/eval_report_pred_confusion.html', context)


def eval_pred_report_confusion_proportion(request):
    '''
        Since the overview page would go first,
        Overview page would create the confusion matrix proportion png in the static folder
        Call it in the html directly
    :param request:
    :return: render request to the confusion matrix page
    '''
    context = {'title': 'Report with a Prediction File'}
    return render(request, 'quality_platform/eval_report_pred_confusion_proportion.html', context)


def eval_pred_report_threshold(request):
    '''

    :param request:
    :return:
    '''
    context = {'title': 'Report with a Prediction File'}
    context['threshold'] = EVAL_REPORT.threshold
    context['threshold_list'] = EVAL_REPORT.threshold_list
    context['threshold_accuracy'] = EVAL_REPORT.threshold_accuracy
    return render(request, 'quality_platform/eval_report_pred_threshold.html', context)


def eval_pred_report_error(request):
    '''

    :param request:
    :return:
    '''
    context = {'title': 'Report with a Prediction File'}
    context['error'] = EVAL_REPORT.error
    return render(request, 'quality_platform/eval_report_pred_error.html', context)