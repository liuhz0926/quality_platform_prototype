import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

class Error_analysis:
    def __init__(self,tfile, pfile, pfile2=None, id_col = 0, r_col=1, p_col=2): #tfile for truth file and pfile for prediction file, r_col stands for
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
        if pfile2:
            self.pfile2 = pd.read_csv(pfile2, sep=self.pd, header=None)
        else:
            self.pfile2 = None
        self.id_col = id_col
        self.r_col = r_col
        self.p_col = p_col
    # def gen_errors(self,file = "tfile"): # if entered truth file we will give instance ID and content. if entered prediction file
    #                             # we will give isntance ID and probability
    #     if file == "tfile":
    #         rfile = self.tfile
    #         cfile = self.pfile
    #     elif file == "pfile":
    #         rfile = self.pfile
    #         cfile = self.tfile
    #     # rfile for result file and cfile for compare file
    def gen_errors(self):

        column = self.tfile.iloc[:, self.r_col]
        misclassified = np.where(column != self.pfile.iloc[:, self.r_col])
        if self.pfile2:
            mis_np = np.concatenate((np.vstack(self.tfile.iloc[:, self.id_col].to_numpy()[misclassified]),
                                     np.vstack(self.tfile.iloc[:, self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile.iloc[:, self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile.iloc[:, self.p_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile2.iloc[:, self.r_col].to_numpy()[misclassified]),
                                     np.vstack(self.pfile2.iloc[:, self.p_col].to_numpy()[misclassified]),
                                     np.vstack(self.tfile.iloc[:, self.p_col].to_numpy()[misclassified])), axis=1)
            df = pd.DataFrame(data=mis_np, columns=["id", "actual", "predicted","probability","predicted_2","probability_2","content"])
        else:
            mis_np = np.concatenate((np.vstack(self.tfile.iloc[:,self.id_col].to_numpy()[misclassified]), np.vstack(self.tfile.iloc[:,self.r_col].to_numpy()[misclassified]), np.vstack(self.pfile.iloc[:,self.r_col].to_numpy()[misclassified]),np.vstack(self.pfile.iloc[:,self.p_col].to_numpy()[misclassified]),np.vstack(self.tfile.iloc[:,self.p_col].to_numpy()[misclassified])),axis=1)
            df = pd.DataFrame(data=mis_np, columns=["id", "actual", "predicted","probability","content"])
        df = df.sort_values(by=['actual'])
        error_data = list(df.T.to_dict().values())
        #print(df)
        return error_data

#test = Error_analysis('2.truth.tsv','3.prediction.tsv')
#test.gen_erros()


