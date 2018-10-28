import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import timeit

#default C=1, default kernel=linear
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

start = timeit.default_timer()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test);
stop = timeit.default_timer()

print("*****CENSUS BASELINE DATA, kernel=linear*****")
print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Time: ", stop - start)
start = timeit.default_timer()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test);
stop = timeit.default_timer()

print("*****CENSUS BASELINE DATA, kernel=rbf*****")
print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Time: ", stop - start)
