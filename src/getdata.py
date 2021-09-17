import pandas as pd
import numpy as np
import os
import requests
from io import StringIO
def get_data():
    data_link='https://raw.githubusercontent.com/Biswajit7890/MLFLOW-Deployment/main/train_LZdllcl.csv'
    read_data=requests.get(data_link).content
    data=pd.read_csv(StringIO(read_data.decode('utf-8')))
    return data



