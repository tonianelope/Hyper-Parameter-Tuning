import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import timeit

class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')
        
#code for importing and pre processing data from https://www.kaggle.com/bananuhbeatdown/multiple-ml-techniques-and-analysis-of-dataset
#dataset itself a truncated version of dataset from https://www.kaggle.com/uciml/adult-census-income
path = 'census_data.csv'
data = pd.read_csv(path)
data = data[data.occupation != '?']
raw_data = data[data.occupation != '?']

print("*****CENSUS BAYES*****")
data['workclass_num'] = data.workclass.map({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
data['over50K'] = np.where(data.income == '<=50K', 0, 1)
data['marital_num'] = data['marital.status'].map({'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3, 'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})
data['race_num'] = data.race.map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
data['rel_num'] = data.relationship.map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
data.head()
X = data[['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain', 'capital.loss']]
y = data.over50K

start = timeit.default_timer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

opt = BayesSearchCV(
    SVC(),
    {
        'C': Real(0.001, 10, prior='log-uniform'),
        'gamma': Real(0.001, 1, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf']),
    },
    n_iter=40
)
opt.fit(X_train, y_train)
stop = timeit.default_timer()

print(opt.score(X_test, y_test))
print(opt.best_params_)
print('Time: ', stop - start)  
