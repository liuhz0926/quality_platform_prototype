import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score

class Threshold_analysis:
    def __init__(self,tfile=None, pfile=None, pfile1 = None,r_col=1, p_col=2, labels = None, predict=False, pretrain=False):
        if predict:
            self.__predict_init__(tfile, pfile, pfile1, r_col, p_col)
        if pretrain:
            self.__pretrain_init__(pfile, labels, r_col, p_col)
        #print(self.pfile)
        #print(self.tfile)


    def __predict_init__(self,tfile, pfile, pfile1 = None,r_col=1, p_col=2): #tfile for truth file and pfile for prediction file, r_col stands for
                                                    # result/class column, p_col stands for column of probablity
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
        self.two_model = False
        if pfile1:
            self.pfile1 = pd.read_csv(pfile1,sep=self.pd,header=None)
            self.two_model = True
        self.r_col = r_col
        self.p_col = p_col

    def __pretrain_init__(self, pfile, labels, r_col=1, p_col=2):
        self.r_col = r_col
        self.p_col = p_col
        self.two_model = False
        predict_df = pd.read_csv(pfile, sep='\t', header=0)
        self.pfile = pd.DataFrame()
        self.pfile['predicted_label'] = predict_df['predicted_label']
        self.pfile['probability'] = predict_df[labels].max(axis=1)
        self.pfile['id'] = self.pfile.index + 1
        self.pfile = self.pfile[['id','predicted_label','probability']]
        #p_file['probability'] = predict_df[["", "B"]].max(axis=1)
        self.tfile = pd.DataFrame()
        self.tfile['id'] = self.pfile['id']
        self.tfile['label'] = predict_df['label']
        self.tfile['content'] = predict_df['text']



    def generate_theshold(self):
        if self.two_model:
            data = pd.concat([self.tfile.iloc[:,self.r_col],self.pfile.iloc[:, self.r_col],self.pfile.iloc[:, self.p_col],self.pfile1.iloc[:, self.r_col],self.pfile1.iloc[:, self.p_col]],axis=1)
        else:
            data = pd.concat([self.tfile.iloc[:,self.r_col],self.pfile.iloc[:, self.r_col],self.pfile.iloc[:, self.p_col]],axis=1)
        total = len(data.index)
        result = []
        threshold_list = []
        accuracy_list = []
        accuracy_list1 = []
        column = self.tfile.iloc[:,self.r_col]
        labels = column.value_counts().keys()
        labels = int(1/len(labels)*100)
        firstrun = True
        for i in range(labels,96,5):
            if i%5 != 0 and not firstrun:
                i += 5-i%5
            if firstrun:
                firstrun = False
            threshold = i / 100
            temp = data[data.iloc[:,2]>i/100]
            instance = len(temp.index)
            percent = len(temp.index)/total*100
            accuracy = accuracy_score(temp.iloc[:,0],temp.iloc[:,1])

            threshold_list.append(threshold)
            accuracy_list.append(round(accuracy, 3))
            if self.two_model:
                temp1 = data[data.iloc[:, 4] > i / 100]
                instance1 = len(temp1.index)
                percent1 = len(temp1.index)/total*100
                accuracy1 = accuracy_score(temp1.iloc[:,0],temp1.iloc[:,3])
                accuracy_list1.append(round(accuracy1,3))
                result.append([threshold, instance, instance1, round(percent, 3), round(percent1, 3),round(accuracy, 3), round(accuracy1,3)])
                #return result, threshold_list, accuracy_list,accuracy_list1
            else:
                result.append([threshold, instance, round(percent, 3), round(accuracy, 3)])
            # result.append("Threshold at "+str(i/100)+": total instance = "+ str(len(temp.index)) + " ("+str(round(percent, 3))+"%), accuracy = " + str(round(accuracy, 3)))
        print(result)
        print(threshold_list)
        print(accuracy_list)
        if self.two_model:
            return result, threshold_list, accuracy_list, accuracy_list1
        return result, threshold_list, accuracy_list

    def generate_roc_curve(self,pos_label): # this is for binary class, and pos_label means input the label for positive class
        actual = self.pfile.iloc[:, self.r_col]
        predict = self.pfile.iloc[:, self.p_col]
        fpr, tpr, thresholds = roc_curve(actual,predict, pos_label=pos_label)
        plt.plot(fpr,tpr, label = 'ROC curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
    #def generate_auc_curve(self):str(len(temp.index))
        #column = self.pfile.iloc[:,self.r_col]
        #labels = column.value_counts().keys()

#test = Threshold_analysis('2.truth.tsv','3.prediction.tsv')
#test.generate_theshold()
#test.generate_roc_curve('yes')
#test.generate_auc_curve()
pfile = 'http://symanto-pastaepizza.northeurope.cloudapp.azure.com:8000/predictions/bpPTl9tKooV1dqeGgL6jUQ.predictions'
labels = ['positive', 'negative']
test = Threshold_analysis(pfile=pfile, labels=labels, pretrain=True)
test.generate_theshold()