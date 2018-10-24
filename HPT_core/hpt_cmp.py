import matplotlib.pyplot as plt
import numpy as np

from time import time
from tqdm.auto import tqdm
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


MODEL = 'Model'
HPT_METHOD = 'Hyper optimization method'
TEST_ACC = 'Test accuracy'
BEST_PARAMS = 'Best Parameters'
CV_TIME = 'Cross validation time (in s)'
PARAMS_SAMPLED = 'Parameters sampled'
MEAN = 'Mean '

DS_SPLITS = 5

def test_func():
    print('Hello3')

def tune_model_cv(X, y, model, hpt_obj, loss, metric, dataset_folds):
    data = {
        MODEL : model.__name__,
        HPT_METHOD : hpt_obj.name,
        TEST_ACC : [],
        BEST_PARAMS : [],
        PARAMS_SAMPLED : [],
        CV_TIME : []
    }
    print(dataset_folds)
    # run outer cross-validation
    with tqdm(total=DS_SPLITS) as pbar:
        for train_i, test_i in dataset_folds:
            X_train, X_test = X[train_i], X[test_i]
            y_train, y_test = y[train_i], y[test_i]

            start = time()
            # for each hyperparam setting run inner cv
            tune_results = hpt_obj.cv(X_train, y_train, model, hpt_obj.param_grid, loss)
            duration = time() - start

            data[CV_TIME].append(duration)
            data[PARAMS_SAMPLED].append(len(tune_results))
            #print(tune_results)
            best_params = sorted(tune_results, key=lambda d: d['loss'])
            best_params = best_params[0]['params']
            data[BEST_PARAMS].append(best_params)

            best_model = model(**best_params)
            best_model.fit(X_train, y_train)
            data[TEST_ACC].append(metric(y_test, best_model.predict(X_test)))
            pbar.update(1)

    # get Mean values
    for item in [TEST_ACC, CV_TIME, PARAMS_SAMPLED]:
        data[MEAN+item] = np.mean(data[item])

    return data

def cmp_hpt_methods(htp_objs, model, dataset, loss, metric, dataset_split=3, name=None):
    print('Start')
    X, y = dataset
    skf = StratifiedKFold(n_splits=5, random_state=dataset_split)
    ds_folds = skf.split(X, y)

    htp_results = []

    print('Here')
    for htp_obj in htp_objs:
        print("HTP using {}".format(htp_obj.name))

        result = tune_model_cv(X, y, model, htp_obj, loss, metric, ds_folds)
        htp_results.append(result)

    return htp_results

def grid_search(X, y, model, param_grid, loss, max_iter=100):
    gs_model = GridSearchCV(model(), param_grid, cv=DS_SPLITS)
    gs_model.fit(X, y)
    obj = {'loss': gs_model.cv_results_['mean_test_score'] * -1, 'params': gs_model.cv_results_['params']}
    return [dict(zip(obj,t)) for t in zip(*obj.values())]
