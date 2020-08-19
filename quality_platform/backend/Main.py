import pandas as pd
import numpy as np
import json
from .evaluate.Overview import Overview
from .evaluate.Threshold_analysis import Threshold_analysis
from .evaluate.Error_analysis import Error_analysis


class Eval_Report:
    def __init__(self):
        # Mark if it is for predict file or pretrain file
        self.predict = None
        self.pretrain = None

        # Overview properties
        self.total_instance = None
        self.evaluate_table = None
        self.instance_class = None

        # Confusion Matrix properties
        self.confusion_labels = None
        self.confusion_data = None
        self.normal_labels = None
        self.normal_data = None

        # Threshold Analysis properties
        self.threshold = None
        self.threshold_accuracy = None
        self.threshold_list = None

        # Error Analysis Properties
        self.error = None

        # Upload another predict file properties
        self.add_total_instance = None
        self.add_evaluate_table = None
        self.add_instance_class = None
        self.add_confusion_labels = None
        self.add_confusion_data = None
        self.add_normal_labels = None
        self.add_normal_data = None
        self.add_threshold_accuracy = None


    def load_report(self, truth_file, prediction_file, add_pred_file = None, labels = None):
        # overview page and confusion matrix
        overview_list, confusion_list = self.load_overview(truth_file, prediction_file, labels = labels, predict=self.predict, pretrain=self.pretrain)
        self.total_instance = overview_list[0]
        self.evaluate_table = overview_list[1]
        self.instance_class = overview_list[2]
        self.confusion_labels = confusion_list[0]
        self.confusion_data = confusion_list[1]
        self.normal_labels = confusion_list[2]
        self.normal_data = confusion_list[3]


        self.load_threshold(truth_file, prediction_file, add_pred_file, labels = labels, predict=self.predict, pretrain=self.pretrain)
        self.load_error(truth_file, prediction_file, add_pred_file, labels = labels, predict=self.predict, pretrain=self.pretrain)
        if add_pred_file:
            add_overview_list, add_confusion_list = self.load_overview(truth_file, add_pred_file, labels = labels, predict=self.predict, pretrain=self.pretrain)
            self.add_total_instance = add_overview_list[0]
            self.add_evaluate_table = add_overview_list[1]
            self.add_instance_class = add_overview_list[2]
            self.add_confusion_labels = add_confusion_list[0]
            self.add_confusion_data = add_confusion_list[1]
            self.add_normal_labels = add_confusion_list[2]
            self.add_normal_data = add_confusion_list[3]

        return

    def load_overview(self, truth_file, prediction_file, labels = None, predict = False, pretrain = False):
        '''

        :param truth_file: tsv file
        :param prediction_file: tsv file
        :return: call overview class to set up evaluation overview report
        '''
        overview = Overview(truth_file, prediction_file, 1,labels=labels, predict = predict, pretrain = pretrain)
        total_instance = overview.total_instance('Truth')
        eval_dict = overview.evaluation()
        evaluate_table = make_eval_table(eval_dict, total_instance)
        instance_class = overview.instance_per_class('Truth')
        overview_list = [total_instance, evaluate_table, instance_class]

        # make confusion matrix
        confusion_labels, confusion_data = overview.confusion_matrix()
        normal_labels, normal_data = overview.confusion_matrix(normalize=True)
        confusion_list = [confusion_labels, confusion_data, normal_labels, normal_data]

        return overview_list, confusion_list

    def load_threshold(self, truth_file, prediction_file, add_pred_file = None, labels = None, predict = False, pretrain = False):
        '''

        :param truth_file: tsv file
        :param prediction_file: tsv file
        :return: call threshold analysis class
        '''
        if add_pred_file:
            threshold_analysis = Threshold_analysis(truth_file, prediction_file, add_pred_file, predict=predict, pretrain=pretrain)
            self.threshold, self.threshold_list, self.threshold_accuracy, self.add_threshold_accuracy = \
                threshold_analysis.generate_theshold()
        else:
            threshold_analysis = Threshold_analysis(truth_file, prediction_file, labels = labels, predict=predict, pretrain=pretrain)
            self.threshold, self.threshold_list, self.threshold_accuracy = threshold_analysis.generate_theshold()

        return

    def load_error(self, truth_file, prediction_file, add_pred_file = None, labels = None, predict = False, pretrain = False):
        '''

        :param truth_file:
        :param prediction_file:
        :return: call error analysis
        '''
        error_analysis = Error_analysis(truth_file, prediction_file, add_pred_file, labels = labels, predict = predict, pretrain = pretrain)
        self.error = error_analysis.gen_errors()

        return


# helper functions:
def make_eval_table(eval_dict, total_instance):
    '''
        make a list of lists of each row in the evaluation table
    :param eval_dict: dict of dict
           total_instance: int
    :return: list of list
    '''
    # init by the key
    eval_table = [[key] for key in eval_dict.keys()]

    for row in eval_table:
        key = row[0]
        if key == 'accuracy':
            row += ['', ''] + [round(eval_dict[key], 3)] + [total_instance]
        else:
            for _, val in eval_dict[key].items():
                row += [round(val, 3)]

    return eval_table
