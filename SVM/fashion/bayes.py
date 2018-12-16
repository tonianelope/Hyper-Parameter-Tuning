import pandas as pd
import numpy as np
import mnist_reader
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import timeit

class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train[:10000]
y_train = y_train[:10000]

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
opt.fit(X_train, y_train)
stop = timeit.default_timer()

print("Bayes on fashion data with kernel poly")
print(opt.score(X_test, y_test))
print(opt.best_params_)
print('Time: ', stop - start)  
