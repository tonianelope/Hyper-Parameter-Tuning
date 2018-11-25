from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from skopt import BayesSearchCV


# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')

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
