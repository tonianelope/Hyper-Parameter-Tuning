from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from sklearn.svm import SVC
from skopt.space import Categorical, Real
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
import timeit
import mnist_reader

#base code from here: https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce, changed$

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train[:10000]
y_train = y_train[:10000]
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

start = timeit.default_timer()

def hyperopt_train_test(params):
    clf = SVC(**params)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    return scores.mean()

space4svm = {
    'C': hp.uniform('C', 0.001, 10),
    'kernel': hp.choice('kernel', ['poly']),
    'gamma': hp.uniform('gamma', 0.001, 1),
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
  
start = timeit.default_timer()

def hyperopt_train_test(params):
    clf = SVC(**params)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    return scores.mean()

space4svm = {
    'C': hp.uniform('C', 0.001, 10),
    'kernel': hp.choice('kernel', ['poly']),
    'gamma': hp.uniform('gamma', 0.001, 1),
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
stop = timeit.default_timer()
print ('best: ', best, ' time: ', stop-start)

