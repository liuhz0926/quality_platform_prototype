import pandas as pd
import numpy as np



class Error_analysis:
    def __init__(self, tfile, pfile, pfile2=None, id_col = 0, r_col=1, p_col=2, labels = None, predict=False, pretrain=False):
        '''
        Create truth file dataframe (id label content) and predit file dataframe (id predict_label probablity)

        :param tfile: truth file
        :param pfile: predicted file
        :param pfile1: addition predicted file
        :param r_col: int, result/class column,  constant set for pandas
        :param p_col: int, column of probablity, constant set for pandas
        :param labels: list of classification labels
        :param predict: boolean
        :param pretrain: boolean
        '''
        if predict:
            self.__predict_init__(tfile, pfile, pfile2=pfile2, id_col = id_col, r_col=r_col, p_col=p_col)
        if pretrain:
            self.__pretrain_init__(pfile, labels, id_col, r_col, p_col)


    def __predict_init__(self,tfile, pfile, pfile2=None, id_col = 0, r_col=1, p_col=2):
        '''

        :param tfile: truth file
        :param pfile: predicted file
        :param pfile1: addition predicted file
        :param r_col: int, result/class column,  constant set for pandas
        :param p_col: int, column of probablity, constant set for pandas
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
        '''

        :param pfile: predicted file, the url coco return
        :param r_col: int, result/class column,  constant set for pandas
        :param p_col: int, column of probablity, constant set for pandas
        :param labels: list of classification labels
        '''
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
        self.tfile = pd.DataFrame()
        self.tfile['id'] = self.pfile['id']
        self.tfile['label'] = predict_df['label']
        self.tfile['content'] = predict_df['text']


    def gen_errors(self):
        '''
        Generate Error Analysis
        The error table contains: ID, Actual Class, Predicted Class, Probability, Content
        If there is another predicetd file, it would contain another predicted label and its probability
        :return: the error table
        '''
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



