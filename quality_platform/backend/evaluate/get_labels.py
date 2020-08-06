import pandas as pd
def get_labels(filename):
    file = pd.read_csv(filename,sep='\t',header=None)
    print(file.iloc[:,1].value_counts().keys().tolist())


