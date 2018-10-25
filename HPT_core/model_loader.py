
from collections import namedtuple
from hyperopt import hp
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

HPT_Model = namedtuple("HPT_Model", "model param_grid hyperopt_grid bayes_grid")

def load_MLPClassifier():
    # hyper parameter names
    lr = 'learning_rate'
    lr_init = 'learning_rate_init'

    # hyper parameter options
    lr_space = [0.01, 0.3, 0.5]
    lr_types = ('constant', 'invscaling', 'adaptive')

    param_grid = {
        #'activation': ('relu', 'tanh', 'logistic'),
        #'solver': ('sgd', 'adam'),
        #'hidden_layer_sizes'
        lr: lr_types,
        lr_init: lr_space
    }

    hyperopt_grid = {
        lr : hp.choice(lr, lr_types),
        lr_init : hp.choice(lr_init, lr_space)
    }

    bayes_grid = {
        lr: Categorical(lr_types),
        lr_init : Real(lr_space[0], lr_space[-1], 'log-uniform')
    }

    return HPT_Model(MLPClassifier, param_grid, hyperopt_grid, bayes_grid)

def load_SVC():
    param_grid = {
        'C':[0.001,0.01,0.1,1,10],
        'gamma': [0.001, 0.01, 0.1, 1.0],
        'kernel': ['linear', 'rbf']
    }

    bayes_grid ={
        'C': Real(0.001, 10, 'log-uniform'),
        'gamma': Real(0.001, 1, 'log-uniform'),
        'kernel': Categorical(['linear', 'rbf'])
    }

    hyperopt_grid = {
        'C': hp.choice('C', [0.001,0.01,0.1,1,10]),
        'gamma': hp.choice('gamma',[0.001, 0.01, 0.1, 1.0]),
        'kernel': hp.choice('kernel', ['linear', 'rbf'])
    }

    return HPT_Model(SVC, param_grid, hyperopt_grid, bayes_grid)

def load_RandomForestClassifier():
    param_grid = {
        "max_depth": [3, None],
        "max_features": [1, 3, 10],
        "min_samples_split": [2, 3, 10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }

    hyperopt_grid = {
        "max_depth": hp.choice("max_depth",[3, None]),
        "max_features": hp.choice("max_features",[1, 3, 10]),
        "min_samples_split": hp.choice("min_samples_split",[2, 3, 10]),
        "bootstrap": hp.choice("bootstrap",[True, False]),
        "criterion": hp.choice("criterion",["gini", "entropy"])
    }

    param_grid = {
        "max_depth": (3, None),
        "max_features": (1, 10),
        "min_samples_split": (2, 10),
        "bootstrap": Categorical(True, False),
        "criterion": Categorical(["gini", "entropy"])
    }

    return HPT_Model(SVC, param_grid, hyperopt_grid, bayes_grid)

