import pandas as pd
import numpy as np
import os

from sklearn.datasets import *
from sklearn.datasets.base import Bunch

def load_census_50k(return_X_y=False):
    filename = 'census_data.csv'
    path = os.path.join('../SVM/census',filename)
    data = pd.read_csv(path)
    data = data[data.occupation != '?']

    data['workclass_num'] = data.workclass.map({
        'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3,
        'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
    data['over50K'] = np.where(data.income == '<=50K', 0, 1)
    data['marital_num'] = data['marital.status'].map({
        'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3,
        'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})
    data['race_num'] = data.race.map({
        'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
    data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
    data['rel_num'] = data.relationship.map(
        {'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
    data.head()

    feature_names = ['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain', 'capital.loss']

    X = np.array(data[feature_names])
    y = np.array(data.over50K)

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y,
                 target_names=['income <=50k', 'income >50k'],
                 DESCR=None,
                 feature_names=feature_names,
                 filename=path)
