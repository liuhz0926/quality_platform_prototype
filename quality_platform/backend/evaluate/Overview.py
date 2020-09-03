import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class Overview:
    def __init__(self,tfile=None, pfile=None, r_col=1, labels = None, predict=False, pretrain=False):
        '''
        Create truth file dataframe (id label content) and predit file dataframe (id predict_label probablity)

        :param tfile: truth file
        :param pfile: predicted file
        :param pfile1: addition predicted file
        :param r_col: int, result/class column,  constant set for pandas
        :param labels: list of classification labels
        :param predict: boolean
        :param pretrain: boolean
        '''
        if predict:
            self.__predict_init__(tfile, pfile, r_col)
        if pretrain:
            self.__pretrain_init__(pfile, r_col, labels)


    def __predict_init__(self, tfile, pfile, r_col):
        '''

        :param tfile: truth file
        :param pfile: predicted file
        :param pfile1: addition predicted file
        :param r_col: int, result/class column,  constant set for pandas
        '''
        if tfile[-3:] == "tsv":
            self.td = '\t' #td means delimeter for tfile
        elif tfile[-3:] == "csv":
            self.td = ','
        else:
            print('Wrong File Type')
            return
        if pfile[-3:] == "tsv":
            self.pd = '\t' #pd means delimeter for pfile
        elif pfile[-3:] == "csv":
            self.pd = ','
        else:
            print('Wrong File Type')
            return
        self.tfile = pd.read_csv(tfile,sep=self.td,header=None)

        self.pfile = pd.read_csv(pfile,sep=self.pd,header=None)
        self.r_col = r_col

    def __pretrain_init__(self, pfile, r_col, labels):
        '''

        :param pfile: predicted file, the url coco return
        :param r_col: int, result/class column,  constant set for pandas
        :param labels: list of classification labels
        '''
        self.r_col = r_col
        predict_df = pd.read_csv(pfile, sep='\t', header=0)
        self.pfile = pd.DataFrame()
        self.pfile['predicted_label'] = predict_df['predicted_label']
        self.pfile['probability'] = predict_df[labels].max(axis=1)
        self.pfile['id'] = self.pfile.index + 1
        self.pfile = self.pfile[['id','predicted_label','probability']]
        self.tfile = pd.DataFrame()
        self.tfile['id'] = self.pfile['id']
        self.tfile['label'] = predict_df['label']
        self.tfile['content'] = predict_df['text']





    def total_instance(self,type):
        '''
        Calculate the number of instance in total
        :param type: truth or prediction
        :return: the number of instance in total
        '''
        if type == 'Truth':
            file = self.tfile
        elif type == 'Prediction':
            file = self.pfile
        else:
            print('Wrong File Type')
            return

        return file.shape[0]

    def instance_per_class(self,type):
        '''
        Calculate the number of instance per class
        :param type: truth or prediction
        :return: list of str, str is a sentence including the number of instance per class
        '''
        if type == 'Truth':
            file = self.tfile
        elif type == 'Prediction':
            file = self.pfile
        else:
            print('Wrong File Type')
            return
        column = file.iloc[:,self.r_col]
        total = column.shape[0]
        result = []
        valuecount = column.value_counts()

        for label in sorted(valuecount.keys()):
            result.append("Class: " + str(label) + ", Count: " + str(valuecount[label]) + ", Percentage: " + str(round(valuecount[label]/total * 100, 2)) + "%\n")

        return result

    def confusion_matrix(self,normalize=False):
        '''
        Calculate the data for confusion matrix in order to make heap map
        :param normalize: boolean, to set confusion matrix in count or percentage
        :return: labels: a list of labels; result: a list of count/percentage for different labels and predicted labels
        '''

        actual = self.tfile.iloc[:,self.r_col]
        predict = self.pfile.iloc[:,self.r_col]
        labels = sorted(list(actual.unique()))
        cm = confusion_matrix(actual,predict,labels=labels)
        result = []
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    result.append([j, i, round(cm[i][j] * 100, 2)])
        else:
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    result.append([j, i, cm[i][j]])

        return labels, result

    def evaluation(self):
        '''
        Get the classification report from sklearn
        :return: classification report
        '''
        actual = self.tfile.iloc[:, self.r_col]
        predict = self.pfile.iloc[:, self.r_col]
        labels = sorted(actual.unique())
        report = classification_report(actual,predict,target_names=labels,output_dict=True)

        return report



