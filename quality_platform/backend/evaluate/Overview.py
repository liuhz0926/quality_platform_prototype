import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class Overview:
    def __init__(self,tfile, pfile, r_col): #tfile for truth file and pfile for prediction file, r_col stands for result/class column
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
    def total_instance(self,type): # type should be Truth or Prediction
        if type == 'Truth':
            file = self.tfile
        elif type == 'Prediction':
            file = self.pfile
        else:
            print('Wrong File Type')
            return
        #print(file.shape[0])
        return file.shape[0]
    def instance_per_class(self,type): # type should be Truth or Prediction
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
        #print(valuecount)
        for label in valuecount.keys():
            result.append("Class: " + str(label) + ", Count: " + str(valuecount[label]) + ", Percentage: " + str(valuecount[label]/total) + "\n")
        #print(result)
        return result

    def confusion_matrix(self):
        actual = self.tfile.iloc[:,self.r_col]
        predict = self.pfile.iloc[:,self.r_col]
        labels = actual.unique()
        cm = confusion_matrix(actual,predict,labels=labels)
        plot_confusion_matrix(cm,labels,normalize=False)
        plot_confusion_matrix(cm,labels,normalize=True)
    def evaluation(self):
        actual = self.tfile.iloc[:, self.r_col]
        predict = self.pfile.iloc[:, self.r_col]
        labels = actual.unique()
        report = classification_report(actual,predict,target_names=labels,output_dict=True)
        #print(report)
        return report





def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if normalize:
        plt.savefig('quality_platform/static/confusion_matrix_proportion.png')
    else:
        plt.savefig('quality_platform/static/confusion_matrix_raw.png')






#test = Overview('2.truthcopy.tsv','3.predictioncopy.tsv',1)
#test.confusion_matrix()
#test.evaluation()
#test.total_instance('Truth')
#test.instance_per_class('Truth')
