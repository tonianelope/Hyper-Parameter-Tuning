import pandas as pd
import numpy as np
import mnist_reader
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import timeit

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train[:10000]
y_train = y_train[:10000]
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("random search for fashion data with kernel poly ") 

start = timeit.default_timer()
parameters = {'C':[0.001,0.01,0.1,1,10],'gamma':[0.001,0.01,0.1,1]}
svc = svm.SVC('poly')
clf = RandomizedSearchCV(svc,parameters,n_iter=40)
clf = svm.SVC(kernel=kernel_name)
scores = cross_val_score(clf, X_train, y_train, cv=5)
clf.fit(X_train,y_train)
print(clf.best_params_)
y_pred = clf.predict(X_test);
stop = timeit.default_timer()

print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
print("Accuracy: (CrossVal) %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Time: ", stop - start)
