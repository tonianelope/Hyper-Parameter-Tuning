import json
from collections import namedtuple
from time import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from tqdm import tqdm

MODEL = 'Model'
HPT_METHOD = 'HPT method'
TEST_ACC = 'Test Accuracy'
TEST_ERR = 'Test Error'
BEST_PARAMS = 'Best Parameters'
CV_TIME = 'Cross-val. time (in s)'
FIT_TIME = 'Fit time'
PARAMS_SAMPLED = 'Parameters sampled'
STD_TEST_SCR = 'Std test score'
MEAN = 'Mean '
INNER_RES = 'Inner result'
CONF_MATRIX = 'Confusion matrix'
ITER = 'Iteration'
ITER_DATA = 'Iteration data'

OUT_DIR = 'output'


HPT_OBJ = namedtuple("HPT_OBJ", 'name param_grid method args')


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
def get_best_params_meanscore(res, score, max_i):
    # argfunc = np.argmin if score_type == 'loss' else np.argmax
    score_type = next(iter(score)) if isinstance(score, dict) else 'score'
    argfunc = np.argmax
    try:
        best_params_index = argfunc(res['mean_test_'+score_type][:max_i])
        std_score = np.array(res['mean_test_'+score_type][:max_i]).mean()
    except Exception as e:
        best_params_index = argfunc(res['mean_test_score'][:max_i])
        std_score = np.array(res['mean_test_score'][:max_i]).mean()
    return best_params_index, std_score

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

            best_params_index, std_score = get_best_params_meanscore(tune_results, score)
            data[STD_TEST_SCR].append(std_score)
            best_params = tune_results['params'][best_params_index]
            data[BEST_PARAMS].append(best_params)

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
def cmp_hpt_methods(dataset, hpt_objs, model, score, final_metric, iters, random_state=3, name=None, verbose=0):
    #X, y =dataset
    X_train, X_test, y_train, y_test = dataset#train_test_split(X, y, test_size=0.2, random_state=random_state)
    results = []

    with tqdm(total=len(hpt_objs)) as pbar:
        for (m_name, param_grid, method, args) in hpt_objs:

            start = time()
            res = method(X_train, y_train, model, param_grid, scoring=score, **args)
            cv_time = time()-start

            data = {
                HPT_METHOD : m_name,
                MODEL : model.__name__,
                INNER_RES : res,
                ITER_DATA : [],
                CV_TIME: cv_time,
            }

            for i in iters:

                best_params_index, std_score = get_best_params_meanscore(res, score, i)
                best_model = model(**res['params'][best_params_index])
                y_pred = best_model.fit(X_train, y_train).predict(X_test)
                acc = accuracy_score(y_test, y_pred)


                data[ITER_DATA].append({
                    ITER : i,
                    BEST_PARAMS : res['params'][best_params_index],
                    CONF_MATRIX : confusion_matrix(y_test, y_pred),
                    TEST_ACC : acc,
                    TEST_ERR : 1.0-acc,
                    STD_TEST_SCR : std_score,
                    PARAMS_SAMPLED : i,
                }
                )

            results.append(data)

            with open('./{}/{}-{}.json'.format(OUT_DIR, name, m_name), 'w') as outfile:
                json.dump(data, outfile, default=default)
            pbar.update(1)

    return results
