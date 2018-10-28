from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from sklearn.svm import SVC
from skopt.space import Categorical, Real
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
import timeit

#base code from here: https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce, changed to fit my own needs

data = load_breast_cancer()
X, X_test, y, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
start = timeit.default_timer()

def hyperopt_train_test(params)
    clf = SVC(**params)
    return cross_val_score(clf, X, y).mean()

space4svm = {
    'C': hp.uniform('C', 0.001, 10),
    'kernel': hp.choice('kernel', ['linear', 'rbf']),
    'gamma': hp.uniform('gamma', 0.001, 1),
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
stop = timeit.default_timer()
print ('best: ', best, ' time: ', stop-start)
