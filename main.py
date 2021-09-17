import pandas as pd
import numpy as np
import os
import csv
from src.getdata import get_data
from src.preprocess import pre_proc
from src.train_test_split import data_split
from src.train import train
from src.evaluation import evaluate
import joblib


df=get_data()
print(df.head())
Absolute_path='E:/iNeuron Assignment'
df.to_csv(Absolute_path+'/data/file.csv')
procdf=pre_proc('E:/iNeuron Assignment/data/file.csv')
print(procdf.head())
procdf.to_csv(Absolute_path+'/processed/process.csv')
traindf,testdf=data_split('E:/iNeuron Assignment/processed/process.csv')
print(traindf.head())
print(testdf.head())
traindf.to_csv(Absolute_path+'/splitfolder/train.csv')
testdf.to_csv(Absolute_path+'/splitfolder/test.csv')
os.mkdir('artifacts')
model_file='model.pkl'
joblib_file = Absolute_path+'/artifacts/'+model_file
for (root,dirs,files) in os.walk('splitfolder'):
    files = str(files)
    if (files =='train.csv'):
       print(files)
       module = train(Absolute_path+'/splitfolder/'+files)
       joblib.dump(module, joblib_file)
    else:
        test_path=(Absolute_path+'/splitfolder/'+files)
        rf_model=joblib.load(joblib_file)
        metdf=evaluate(test_path,rf_model)
        metdf.to_json(Absolute_path+'/artifacts/metrics.json',orient='records')



















