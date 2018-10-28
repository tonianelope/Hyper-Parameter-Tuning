import pandas as pd
import numpy as np
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from sklearn.svm import SVC
from skopt.space import Categorical, Real
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
import timeit

#base code from here: https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce, changed to fit my own needs

#code for importing and pre processing data from https://www.kaggle.com/bananuhbeatdown/multiple-ml-techniques-and-analysis-of-dataset
#dataset itself a truncated version of dataset from https://www.kaggle.com/uciml/adult-census-income
path = 'census_data.csv'
data = pd.read_csv(path)
data = data[data.occupation != '?']
raw_data = data[data.occupation != '?']

data['workclass_num'] = data.workclass.map({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
data['over50K'] = np.where(data.income == '<=50K', 0, 1)
data['marital_num'] = data['marital.status'].map({'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3, 'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})
data['race_num'] = data.race.map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
data['rel_num'] = data.relationship.map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
data.head()
X = data[['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain', 'capital.loss']]
y = data.over50K
X, X_test, y, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
start = timeit.default_timer()

def hyperopt_train_test(params):
    clf = SVC(**params)
    return cross_val_score(clf, X, y).mean()

space4svm = {
    'C': hp.loguniform('C', np.log(0.001), np.log(100)),
    'kernel': hp.choice('kernel', ['linear', 'rbf']),
    'gamma': hp.loguniform('gamma', np.log(0.001), np.log(1)),
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=10, trials=trials)
stop = timeit.default_timer()
print ('best: ', best, ' time: ', stop-start)
