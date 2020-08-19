import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

class Error_analysis:
    def __init__(self, tfile, pfile, pfile2=None, id_col = 0, r_col=1, p_col=2, labels = None, predict=False, pretrain=False):

        if predict:
            self.__predict_init__(tfile, pfile, pfile2=pfile2, id_col = id_col, r_col=r_col, p_col=p_col)
        if pretrain:
            self.__pretrain_init__(pfile, labels, id_col, r_col, p_col)


    def __predict_init__(self,tfile, pfile, pfile2=None, id_col = 0, r_col=1, p_col=2): #tfile for truth file and pfile for prediction file, r_col stands for
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
        self.second = False
        if pfile2:
            self.second = True
            self.pfile2 = pd.read_csv(pfile2, sep=self.pd, header=None)
        else:
            self.pfile2 = None
        self.id_col = id_col
        self.r_col = r_col
        self.p_col = p_col

    def __pretrain_init__(self, pfile, labels, id_col = 0, r_col=1, p_col=2):
        self.id_col = id_col
        self.r_col = r_col
        self.p_col = p_col
        self.second = False
        self.pfile2 = None
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


    def gen_errors(self):

        column = self.tfile.iloc[:, self.r_col]
        misclassified = np.where(column != self.pfile.iloc[:, self.r_col])
        if self.second:
            mis_np = np.concatenate((np.vstack(self.tfile.iloc[:, self.id_col].to_numpy()[misclassified]),
                                     np.vstack(self.tfile.iloc[:, self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile.iloc[:, self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile.iloc[:, self.p_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile2.iloc[:, self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile2.iloc[:, self.p_col].to_numpy()[misclassified]),
                                     np.vstack(self.tfile.iloc[:, self.p_col].to_numpy()[misclassified])), axis=1)
            df = pd.DataFrame(data=mis_np, columns=["id", "actual", "predicted_1","probability_1","predicted_2","probability_2","content"])
        else:
            mis_np = np.concatenate((np.vstack(self.tfile.iloc[:,self.id_col].to_numpy()[misclassified]),
                                     np.vstack(self.tfile.iloc[:,self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile.iloc[:,self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile.iloc[:,self.p_col].to_numpy()[misclassified]),
                                     np.vstack(self.tfile.iloc[:,self.p_col].to_numpy()[misclassified])), axis=1)
            df = pd.DataFrame(data=mis_np, columns=["id", "actual", "predicted","probability","content"])
        df = df.sort_values(by=['actual'])
        error_data = list(df.T.to_dict().values())
        for dict in error_data:
            if self.second:
                dict['probability_1'] = round(float(dict['probability_1']), 5)
                dict['probability_2'] = round(float(dict['probability_2']), 5)
            else:
                dict['probability'] = round(float(dict['probability']), 5)
        return error_data


    # def gen_errors(self,file = "tfile"): # if entered truth file we will give instance ID and content. if entered prediction file
    #                             # we will give isntance ID and probability
    #     if file == "tfile":
    #         rfile = self.tfile
    #         cfile = self.pfile
    #     elif file == "pfile":
    #         rfile = self.pfile
    #         cfile = self.tfile
    #     # rfile for result file and cfile for compare file

#test = Error_analysis('2.truth.tsv','3.prediction.tsv')
#test.gen_erros()


