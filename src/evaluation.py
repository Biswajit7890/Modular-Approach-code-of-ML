import pandas as pd
import numpy as np
#from sklearn.metrics import roc_auc
import pickle
from sklearn.preprocessing import StandardScaler

def evaluate(test_path,model):
    test=pd.read_csv(test_path)
    X=test.drop(labels='is_promoted', axis=1)
    sc_test=StandardScaler()
    X=sc_test.fit_transform(X)
    preds=model.predict(X)
    pred_df=preds.to_df()
    return pred_df



