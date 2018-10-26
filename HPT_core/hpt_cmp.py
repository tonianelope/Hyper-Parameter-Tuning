import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from collections import namedtuple
from dotmap import DotMap
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from time import time
from tqdm import tqdm_notebook as tqdm
#from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV

MODEL = 'Model'
HPT_METHOD = 'HPT method'
TEST_ACC = 'Test accuracy'
BEST_PARAMS = 'Best Parameters'
CV_TIME = 'Cross-val. time (in s)'
PARAMS_SAMPLED = 'Parameters sampled'
STD_TEST_ACC = 'Std'
MEAN = 'Mean '
INNER_RES = 'Inner result'

DEFAULT_COLUMNS = [
    HPT_METHOD, MEAN+TEST_ACC, MEAN+STD_TEST_ACC, MEAN+CV_TIME, MEAN+PARAMS_SAMPLED
]

DS_SPLITS = 2
MAX_ITER = 10

HPT_OBJ = namedtuple("HPT_OBJ", 'name param_grid method args')

# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')

def tune_model_cv(X, y, model, hpt_obj, loss, metric, dataset_folds):
    data = {
        MODEL : model.__name__,
        HPT_METHOD : hpt_obj.name,
        TEST_ACC : [],
        STD_TEST_ACC: [],
        BEST_PARAMS : [],
        PARAMS_SAMPLED : [],
        CV_TIME : [],
        INNER_RES : []
    }

    # run outer cross-validation
    #with tqdm(total=DS_SPLITS) as pbar:
    for train_i, test_i in dataset_folds:
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]
        
        start = time()
        # for each hyperparam setting run inner cv
        tune_results = hpt_obj.method(
            X_train, y_train, model,
            hpt_obj.param_grid,
            scoring=loss,
            **hpt_obj.args
        )
        duration = time() - start
        
        data[CV_TIME].append(duration)
        data[INNER_RES].append(tune_results)
        data[PARAMS_SAMPLED].append(len(tune_results))
        # print(tune_results)
        best_params = sorted(tune_results, key=lambda d: d['mean_test_score'])
        data[STD_TEST_ACC].append(best_params[0]['std'])
        best_params = best_params[0]['params']
        data[BEST_PARAMS].append(best_params)
        # print(best_params)
        best_model = model(**best_params)
        best_model.fit(X_train, y_train)
        data[TEST_ACC].append(metric(y_test, best_model.predict(X_test)))
        #       pbar.update(1)

    # get Mean values
    for item in [TEST_ACC, CV_TIME, PARAMS_SAMPLED, STD_TEST_ACC]:
        data[MEAN+item] = np.mean(data[item])

    return data

def cmp_hpt_methods_double_cv(dataset, hpt_objs, model, loss, metric, random_state=3, name=None, max_iter=0):
    MAX_ITER = max_iter
    print(MAX_ITER)
    X, y = dataset
    skf = StratifiedKFold(n_splits=DS_SPLITS, random_state=random_state)
    ds_folds = skf.split(X, y)

    htp_results = []


    for htp_obj in hpt_objs:
        # print("HTP using {}".format(htp_obj.name))

        result = tune_model_cv(X, y, model, htp_obj, loss, metric, skf.split(X,y))
        result['dataset']=name
        htp_results.append(result)

    return htp_results

def cmp_hpt_methods(dataset, hpt_objs, model, loss, metric, random_state=3, name=None, max_iter=0, verbose=0):
    X, y =dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    results = []

    for (name, param_grid, method, args) in hpt_objs:
        res = method(X_train, y_train, model, param_grid, scoring=loss, verbose=1, **args)

        print(res)
        best_params_index = np.argmax(res['mean_test_score'])
        best_params = res['params'][best_params_index]
        best_model = model(**best_params)
        y_pred = best_model.fit(X_train, y_train).predict(X_test)

        results.append({
            'method':name,
            'data': res,
            'best_params': best_params,
            'conf_matrix': confusion_matrix(y_test, y_pred),
        })
    return results


#-----------HYPERTUNE METHODS--------------------

def mean_results(results, params):
    mean = 'mean_'
    std_ = 'std_'
    tes = 'test_score'
    trs = 'train_score'
    ft = 'fit_time'
    p = 'params'

    cv_results = {}
    for label in [tes, trs, ft]:
        cv_results[mean+label] = results[label].mean()
        cv_results[std_+label] = results[label].std()

    cv_results[p] = params
    for k in params:
        cv_results[p+"_"+k] = params[k]
    return cv_results

def run_cv(X, y, model, params, scoring, verbose=1, max_iter=MAX_ITER):
    m = model(**params)
    scores = cross_validate(m, X, y, scoring=scoring, cv=DS_SPLITS)
    scores = mean_results(scores, params)
    scores['status'] = STATUS_OK
    scores['loss'] = scores['mean_test_score']
    return scores
    # return  {
    #     'loss': -1*scores.mean(),
    #     'params': params,
    #     'std': scores.std(),
    #     'status': STATUS_OK}

def run_baseline(*args, **kargs):
    res = [run_cv(*args, **kargs)]
    return {k: [dic[k] for dic in res] for k in res[0]}

def sklearn_search(X, y, s_model):
    #s_model = sk_search(model(), param_grid, cv=DS_SPLITS, n_iter=max_iter)
    s_model.fit(X, y)
    obj = {
        'loss': -1*np.array(s_model.cv_results_['mean_test_score']),
        'params': s_model.cv_results_['params'],
        'std': s_model.cv_results_['std_test_score']
    }
    # convert obj of lists to list of objs
    return s_model.cv_results_ #[dict(zip(obj,t)) for t in zip(*obj.values())]

def grid_search(X, y, model, param_grid, **kargs):
    gs = GridSearchCV(model(), param_grid, cv=DS_SPLITS, **kargs)
    return sklearn_search(X, y, gs)

def random_search(X, y, model, param_grid, **kargs):
    rs = RandomizedSearchCV(model() , param_grid, cv=DS_SPLITS, **kargs)
    return sklearn_search(X, y, rs)

def baysian_search(X, y, model, params, **kargs):
    bs = BayesSearchCV(model(), params, cv=DS_SPLITS, **kargs)
    return sklearn_search(X, y, bs)

def tpe_search(X, y, model, param_grid, scoring, verbose=1, max_iter=MAX_ITER):
    trials = Trials()
    results = fmin(
        fn=lambda param: run_cv(X, y, model, param, scoring),
        space=param_grid,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_iter
    )
    return {k: [dic[k] for dic in trials.results] for k in trials.results[0]}


#--------VISUALISE/SUMMARY FUNCTIONALITY------------

def table(results, columns=DEFAULT_COLUMNS):
    df = pd.DataFrame(results)
    df = df[columns] # select columns to return
    return df

def table_by_ds(all_results, datasets):
    tables = [table(i) for i in all_results]
    return pd.concat(tables, keys=datasets , axis=1)

def plot_by_ds(val, list_of_results, datasets):
    l = len(datasets)
    y = [i for i in range(l)]
    df = table_by_ds(list_of_results, datasets)
    methods = df[datasets[0]][HPT_METHOD]

    fig, ax = plt.subplots()

    # for each method
    for i, method in enumerate(methods):
        x_axis = df.T.loc[(slice(None), val), :][i]
        ax.plot(y, x_axis, label=method)

    ax.legend()
    ax.set_xticks(y)
    ax.set_xticklabels(datasets, rotation=45)
    ax.set_xlabel('Datasets')
    ax.set_ylabel(val)

def plot_confusion_matrix(cfm, classes, normalise=True, title='Confusion Matrix'):
    if normalise:
        cfm = cfm.astype('float')/ cfm.sum(axis=1)[:,np.newaxis]

    plt.figure()
    ax = sn.heatmap(cfm, annot=True,cmap=plt.cm.Blues) #annot_kws={"size": 16} #font size

    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
