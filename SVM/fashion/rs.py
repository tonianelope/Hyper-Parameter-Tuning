import pandas as pd
import numpy as np
import mnist_reader
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import timeit

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train[:10000]
y_train = y_train[:10000]

print("random search for fashion data with kernel poly ") 

start = timeit.default_timer()
parameters = {'C':[0.001,0.01,0.1,1,10],'gamma':[0.001,0.01,0.1,1]}
svc = svm.SVC('poly')
clf = RandomizedSearchCV(svc,parameters,n_iter=40)
clf.fit(X_train,y_train)
print(clf.best_params_)
y_pred = clf.predict(X_test);
stop = timeit.default_timer()

print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
print("Time: ", stop - start)
