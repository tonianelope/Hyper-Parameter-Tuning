import json
from collections import namedtuple
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from skopt import BayesSearchCV
from tqdm import tqdm

MODEL = 'Model'
HPT_METHOD = 'HPT method'
TEST_ACC = 'Test Accuracy'
TEST_ERR = 'Test Error'
BEST_PARAMS = 'Best Parameters'
CV_TIME = 'Cross-val. time (in s)'
FIT_TIME = 'Fit time'
PARAMS_SAMPLED = 'Parameters sampled'
STD_TEST_SCR = 'Std'
MEAN = 'Mean '
INNER_RES = 'Inner result'
CONF_MATRIX = 'Confusion matrix'

DEFAULT_COLUMNS = [
    HPT_METHOD, MEAN+TEST_ACC, MEAN+STD_TEST_SCR, MEAN+CV_TIME, MEAN+PARAMS_SAMPLED
]

OUT_DIR = '.mlpc_output'
PLOT_DIR = 'plots'

HPT_OBJ = namedtuple("HPT_OBJ", 'name param_grid method args')

# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')


#-------------HELPER FUNCTIONS-----------------------
'''
enable json.dump to convert numpy objs
'''
def default(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
                        np.float64)):
        return float(obj)
    elif isinstance(obj,(np.ndarray,)): #### This is the fix
        return obj.tolist()
    else:
        print("Type: {}, Obj: {}".format(type(obj), obj))
    raise TypeError('Not serializable: {} - {}'.format(obj, type(obj)))


'''
gets the pest parameters, based onthe scoring metric
'''
def get_best_params(res, score):
    # argfunc = np.argmin if score_type == 'loss' else np.argmax
    score_type = next(iter(score)) if isinstance(score, dict) else 'score'
    argfunc = np.argmax
    try:
        best_params_index = argfunc(res['mean_test_'+score_type])
        best_score = np.max(res['mean_test_'+score_type])
    except Exception as e:
        best_params_index = argfunc(res['mean_test_score'])
        best_score = np.max(res['mean_test_score'])
    return best_params_index, best_score

#-------------COMPARE HPT METHODS WITH DOUBLE CROSS VALIDATION-------------
# Double cross validation approach based on the following repo (https://github.com/roamanalytics/roamresearch/tree/master/BlogPosts/Hyperparameter_tuning_comparison)
'''
Run double cross validation on the hpt_obj.
Record the Data
'''
def tune_model_cv(X, y, model, hpt_obj, score, final_metric, cv, dataset_folds):
    data = {
        MODEL : model.__name__,
        HPT_METHOD : hpt_obj.name,
        TEST_ACC : [],
        TEST_ERR : [],
        STD_TEST_SCR: [],
        BEST_PARAMS : [],
        PARAMS_SAMPLED : [],
        CV_TIME : [],
        INNER_RES : [],
        CONF_MATRIX : [],
    }

    with tqdm(total=cv, desc='Inner CV') as pbar:
    # run outer cross-validation
        for train_i, test_i in dataset_folds:
            X_train, X_test = X[train_i], X[test_i]
            y_train, y_test = y[train_i], y[test_i]

            start = time()
            # for each hyperparam setting run inner cv
            tune_results = hpt_obj.method(
                X_train, y_train, model,
                hpt_obj.param_grid,
                scoring=score,
                **hpt_obj.args
            )
            duration = time() - start

            data[CV_TIME].append(duration)
            data[INNER_RES].append(tune_results)
            data[PARAMS_SAMPLED].append(len(tune_results))

            best_params_index, best_score = get_best_params(tune_results, score)
            data[STD_TEST_SCR].append(best_score)
            data[BEST_PARAMS].append(tune_results['params'][best_params_index])

            best_model = model(**best_params)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            acc = final_metric(y_test, y_pred)
            data[TEST_ACC].append(acc)
            data[TEST_ERR].append(1-acc)
            data[CONF_MATRIX].append(confusion_matrix(y_test, y_pred))

            pbar.update(1)

    # get Mean values
    for item in [TEST_ACC, TEST_ERR, CV_TIME, PARAMS_SAMPLED]: #STD_TEST_ACC
        data[MEAN+item] = np.mean(data[item])

    return data

'''
compares the `hpt_objs`, for `model` - using double cross-validation
scoring the tuning on `score` and the final results on `final_metric`
dataset needs to be a tuple of (X,y)
'''
def cmp_hpt_methods_double_cv(dataset, hpt_objs, model, score, final_metric, cv=5, random_state=3, name=None):
    X, y = dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    skf = StratifiedKFold(n_splits=cv, random_state=random_state)
    ds_folds = skf.split(X_train, y_train)

    htp_results = []

    with tqdm(total=len(hpt_objs), desc='Outer Loop') as pbar:
        for htp_obj in hpt_objs:
            result = tune_model_cv(X, y, model, htp_obj, score, final_metric, cv, skf.split(X,y))
            htp_results.append(result)
            with open('./{}/{}-{}.json'.format(OUT_DIR, name, htp_obj.name), 'w') as outfile:
                json.dump(result, outfile, default=default)
            pbar.update(1)

    return htp_results

#--------------COMPARE HYPERTUNE METHODS (SINGLE CROSS VALIDATION)----------------------
'''
compares the `hpt_objs`, for `model` - using double cross-validation
scoring the tuning on `score` and the final results on `final_metric`
dataset needs to be a tuple of (X,y)
'''
def cmp_hpt_methods(dataset, hpt_objs, model, score, final_metric, max_iter, random_state=3, name=None, verbose=0):
    X, y =dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    results = []

    with tqdm(total=len(hpt_objs)) as pbar:
        for (m_name, param_grid, method, args) in hpt_objs:

            start = time()
            res = method(X_train, y_train, model, param_grid, scoring=score, **args)
            cv_time = time()-start

            best_params =get_best_params(res, score)
            best_model = model(**best_params)
            y_pred = best_model.fit(X_train, y_train).predict(X_test)

            data = {
                HPT_METHOD : m_name,
                INNER_RES : res,
                BEST_PARAMS : best_params,
                CONF_MATRIX : confusion_matrix(y_test, y_pred),
                TEST_ACC : accuracy_score(y_test, y_pred),
                PARAMS_SAMPLED : len(res['params']),
                CV_TIME: cv_time,
            }
            results.append(data)

            with open('./{}/{}-{}-{}-{}.json'.format(OUT_DIR, name, m_name), 'w') as outfile:
                json.dump(data, outfile, default=default)
            pbar.update(1)

    return results

#-----------HYPERTUNE METHODS--------------------
'''
compute mean of cv_results
'''
def mean_results(results, params):
    mean = 'mean_'
    std_ = 'std_'
    tes = 'test_'
    trs = 'train_score'
    ft = 'fit_time'
    ps = 'params'
    p = 'param'

    cv_results = {}
    for label in results:
        cv_results[mean+label] = results[label].mean()
        cv_results[std_+label] = results[label].std()

    cv_results[ps] = params
    for k in params:
        cv_results[p+"_"+k] = params[k]
    return cv_results

def run_cv(X, y, model, params, scoring, cv):
    #print(params)
    m = model(**params)
    scores = cross_validate(m, X, y, scoring=scoring, cv=cv, return_train_score=True)

    score_lable = 'mean_test_score'
    if isinstance(scoring, dict):
        score_lable = 'mean_test_loss'
    score_type = next(iter(scoring)) if isinstance(scoring, dict) else 'score'
    scores = mean_results(scores, params)
    scores['status'] = STATUS_OK
    scores['loss'] = -1* scores['mean_test_'+score_type]
    return scores

def run_baseline(*args, **kargs):
    res = [run_cv(*args, **kargs)]
    return {k: [dic[k] for dic in res] for k in res[0]}

def sklearn_search(X, y, s_model):
    s_model.fit(X, y)
    return s_model.cv_results_

def grid_search(X, y, model, param_grid, **kargs):
    gs = GridSearchCV(model(), param_grid, **kargs)
    return sklearn_search(X, y, gs)

def random_search(X, y, model, param_grid, **kargs):
    rs = RandomizedSearchCV(model() , param_grid, **kargs)
    return sklearn_search(X, y, rs)

def baysian_search(X, y, model, params, scoring, **kargs):
    scoring = next(iter(scoring)) if isinstance(scoring, dict) else scoring
    bs = BayesSearchCV(model(), params, **kargs)
    # TODO undo
    r = sklearn_search(X, y, bs)
    print(r)
    return r

def test_tpe(X,y,model,param,scoring,cv):
    m = model(**param)
    err = cross_val_score(m, X, y, scoring=scoring, cv=cv).mean()
    return {'loss': -err, 'status': STATUS_OK}

def plot_tpe_res(trials, params):
    cols = len(params)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(15,5))#, figsize=(10,10))
    cmap = plt.cm.jet
    #print(trials.trials[0]['misc']['vals'])
    for i, key in enumerate(params):
        xs = [t['misc']['vals'][key] for t in trials.trials]
        ys = [-t['result']['loss'] for t in trials.trials]

        #xs, ys = zip(\*sorted(zip(xs, ys)))
        #ys = np.array(ys)w
        if(cols>1):
            axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/cols))
            axes[i].set_title(key)
        else:
            axes.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/cols))
            axes.set_title(key)

        # ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
        # ax.set_title('Iris Dataset - KNN', fontsize=18)
        # ax.set_xlabel('test', fontsize=12)
        # ax.set_ylabel('cross validation accuracy', fontsize=12)

def plot_tpe(trials):
    for t in trials.trials:
        print(t['result']['loss'])

    f, ax = plt.subplots()
    xs = [i for i in range(len(trials.trials))]
    ys = [-t['result']['loss'] for t in trials.trials]
    sn.lineplot(x=xs, y=ys)

def tpe_search(X, y, model, param_grid, scoring, max_iter, cv):
    trials = Trials()
    results = fmin(
        fn=lambda param: run_cv(X, y, model, param, scoring, cv),
        space=param_grid,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_iter
    )
    #plot_tpe_res(trials, param_grid)
    #plot_tpe(trials)
    return {k: [dic[k] for dic in trials.results] for k in trials.results[0]}


#--------VISUALISE/SUMMARY FUNCTIONALITY------------
'''
transform results into pandas DataFrame
'''
def table(results, columns=DEFAULT_COLUMNS):
    df = pd.DataFrame(results)
    df = df[columns] # select columns to return
    return df
'''
transform a list of results into a list of pandas Dataframes
'''
def table_by_ds(all_results, datasets):
    tables = [table(i) for i in all_results]
    return pd.concat(tables, keys=datasets , axis=1)

'''
plot the value of `val` for all method by dataset
'''
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

'''
cfm       -> (numpy array) - confusion matrix
classes   -> (array) - target lables for plot
noramlise -> (bool) - noramlise matrix
title     -> (str) - plot title
'''
def plot_confusion_matrix(cfm, classes, normalise=True, title='Confusion Matrix'):
    if normalise:
        cfm = cfm.astype('float')/ cfm.sum(axis=1)[:,np.newaxis]

    plt.figure()
    ax = sn.heatmap(cfm, annot=True, cmap=plt.cm.Blues)

    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

'''
all_results  -> list of dicts, where INNER_RES contains the method trials
params       -> list of the parameters to plot
scoring      -> name of the column to score them by
param_classes-> dict with lables for parameters
'''
def boxplot_param_distribution(all_results, params, scoring, param_classes=None):

    for param in params:
        plt.figure()
        # plt.xticks(rotation=45)
        for method in all_results:
            #pick x n y
            ax = sn.boxplot(y='param_'+param, x=scoring, data=method[INNER_RES])
            ax.set_ylabel(scoring)
            ax.set_xlabel(param)
'''
all_results  -> list of dicts, where INNER_RES contains the method trial
params       -> list of the parameters to plot
scoring      -> name of the column to score them by
param_classes-> dict with lables for parameters
'''
def scatterplot_param_distribution(all_results, params, scoring, param_classes=None):

    for param in params:
        plt.figure()
        # plt.xticks(rotation=45)
        for method in all_results:
            #pick x n y
            ax = sn.scatterplot(x='param_'+param, y=scoring, data=method[INNER_RES], label=method[HPT_METHOD])
            ax.set_ylabel(scoring)
            ax.set_xlabel(param)

def barplot(x, y, data, textval, xlabel=None, ylabel=None, xtick=[], ytick=[]):
    fig, ax = plt.subplots()
    g = sn.barplot(x=x, y=y, data=data, ax=ax)

    for i, val in enumerate(data.iterrows()):
        px, py = (i, val[1][y]) if x==HPT_METHOD else (val[1][x], i)
        t = round(float(val[1][textval]),2)
        g.text(px, py, t, ha='center')

    g.set_ylabel(ylabel)
    g.set_xlabel(xlabel)

    g.set_yticklabels(ytick)
    g.set_xticklabels(xtick)

    fig.tight_layout()

def saveplot(path):
    plt.savefig(path)
