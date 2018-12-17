import pandas as pd
import numpy as np
import mnist_reader
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
import timeit

class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train[:10000]
y_train = y_train[:10000]
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

start = timeit.default_timer()
opt = BayesSearchCV(
    SVC(),
    {
        'C': Real(0.001, 10, prior='log-uniform'),
        'gamma': Real(0.001, 1, prior='log-uniform'),
        'kernel': Categorical(['poly']),
    },
    n_iter=40
)
scores = cross_val_score(opt, X_train, y_train, cv=5)
opt.fit(X_train, y_train)
stop = timeit.default_timer()

print("Bayes on fashion data with kernel poly")
print(opt.score(X_test, y_test))
print(opt.best_params_)
print("Accuracy: (CrossVal) %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print('Time: ', stop - start)  
