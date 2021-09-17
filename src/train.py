import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
#from skopt import BayesSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



def train(train_path):
    train =pd.read_csv(train_path)
    X=train.drop(labels='is_promoted', axis=1)
    y=train['is_promoted'].values
    sc_train=StandardScaler()
    X=sc_train.fit_transform(X)
    kf = StratifiedKFold(n_splits=3)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nb3 = RandomForestClassifier()
        search = nb3.fit(X_train, y_train)
        pred_test = search.predict(X_test)
        pred_train = search.predict(X_train)
        print("The confusion Matrix", confusion_matrix(y_test, pred_test))
        print("The log_loss of Train", roc_auc_score(y_train, pred_train))
        print("The log_loss of Test", roc_auc_score(y_test, pred_test))
        print('*' * 50)
    return (search)



