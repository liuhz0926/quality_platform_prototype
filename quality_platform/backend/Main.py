import pandas as pd
import numpy as np
import json
from .evaluate.Overview import Overview
from .evaluate.Threshold_analysis import Threshold_analysis
from .evaluate.Error_analysis import Error_analysis


class Eval_Report:
    def __init__(self):
        self.total_instance = None
        self.evaluate_table = None
        self.instance_class = None
        self.threshold = None
        self.threshold_accuracy = None
        self.threshold_list = None
        self.error = None
        self.add_total_instance = None
        self.add_evaluate_table = None
        self.add_instance_class = None
        self.add_threshold_accuracy = None


    def load_report(self, truth_file, prediction_file, add_pred_file = None):
        # overview page and confusion matrix
        self.total_instance, self.evaluate_table, self.instance_class = \
            self.load_overview(truth_file, prediction_file)
        self.load_threshold(truth_file, prediction_file, add_pred_file)
        self.load_error(truth_file, prediction_file)
        if add_pred_file != None:
            self.add_total_instance, self.add_evaluate_table, self.add_instance_class = \
                self.load_overview(truth_file, add_pred_file)
        return

    def load_overview(self, truth_file, prediction_file):
        '''

        :param truth_file: tsv file
        :param prediction_file: tsv file
        :return: call overview class to set up evaluation overview report
        '''
        overview = Overview(truth_file, prediction_file, 1)
        total_instance = overview.total_instance('Truth')
        eval_dict = overview.evaluation()
        evaluate_table = make_eval_table(eval_dict, total_instance)
        instance_class = overview.instance_per_class('Truth')

        # make confusion matrix
        overview.confusion_matrix()

        return total_instance, evaluate_table, instance_class

    def load_threshold(self, truth_file, prediction_file, add_pred_file = None):
        '''

        :param truth_file: tsv file
        :param prediction_file: tsv file
        :return: call threshold analysis class
        '''
        if add_pred_file == None:

            threshold_analysis = Threshold_analysis(truth_file, prediction_file)
            self.threshold, self.threshold_list, self.threshold_accuracy = threshold_analysis.generate_theshold()
            return
        else:
            threshold_analysis = Threshold_analysis(truth_file, prediction_file, add_pred_file)
            self.threshold, self.threshold_list, self.threshold_accuracy, self.add_threshold_accuracy = \
                threshold_analysis.generate_theshold()


        return

    def load_error(self, truth_file, prediction_file):
        '''

        :param truth_file:
        :param prediction_file:
        :return: call error analysis
        '''
        error_analysis = Error_analysis(truth_file, prediction_file)
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