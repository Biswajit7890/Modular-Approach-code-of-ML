import pandas as pd
import numpy as np
from sklearn import preprocessing
def pre_proc(data_path):
    data=pd.read_csv(data_path)
    mode_edu = data['education'].mode()[0]
    mode_prev_year = data['previous_year_rating'].mode()[0]
    data['education'] = data['education'].fillna(mode_edu)
    data['previous_year_rating'] = data['previous_year_rating'].fillna(mode_prev_year)
    le = preprocessing.LabelEncoder()
    data['education'] = le.fit_transform(data['education'])
    data['gender'] = le.fit_transform(data['gender'])
    data['recruitment_channel'] = le.fit_transform(data['recruitment_channel'])
    data['region'] = le.fit_transform(data['region'])
    data['department'] = le.fit_transform(data['department'])
    return data




