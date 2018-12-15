import getopt
import sys
import warnings

import seaborn as sns
from sklearn import (discriminant_analysis, ensemble, gaussian_process,
                     linear_model, naive_bayes, neighbors, neural_network,
                     preprocessing, svm, tree)
from sklearn.metrics import (accuracy_score, f1_score, log_loss, make_scorer,
                             recall_score, roc_auc_score)
from tqdm import tnrange, tqdm

import dataset_loader as ds
import model_loader as mdl
from hpt_cmp import *
from hpt_methods import *
from plots import *
from skopt.space import Categorical, Integer, Real

#warnings.filterwarnings('ignore')


try:
    opts, args = getopt.getopt(sys.argv, "n:")
except getopt.GetoptError:
    print('run_mlcp.py -n <output_name>')
    sys.exit(2)

name = args[1]
st = int(args[2])
end = int(args[3])
step = int(args[4])
max_iter = end

print(name)
print(max_iter)
#sys.exit()

CV_SPLITS = 2

# LOAD DATA
#dsBunch = ds.load('iris')
#data = train_test_split(dsBunch.data, dsBunch.target, test_size=0.25, random_state=1)

dsBunch = ds.load_mnist_back()
dsTest = ds.load_mnist_back_test()

data = (preprocessing.scale(dsBunch.data), dsTest.data, dsBunch.target, dsTest.target)
# #print('DATA:')
n_features = dsBunch.data.shape[1]
shp = dsBunch.data.shape
#print(pd.DataFrame(dsTest.data).head)
#print()
#print(pd.DataFrame(dsTest.target).head)
#print('n_features: {}\nshape: {}\n'.format(n_features, shp))


# DEFINE PARAM GRIDS
d_features = n_features//2
hls = [(d_features,)*3, (n_features,)*3, (d_features,)*2, (n_features,)*2, (d_features,), (n_features,),]
alpha = [0.0001, 0.001, 0.01, 0.1]
lr_init = [0.0001, 0.001, 0.01, 0.1, 1]
lr = ['adaptive']
solver = ['adam', 'sgd']
rs = [1]

# sklean paramgrid
pg = {
    'hidden_layer_sizes': hls,
    'alpha': alpha,
    'learning_rate': lr,
    'learning_rate_init': lr_init,
    'random_state': rs,
    'solver': solver
}

# hyperopt paramgird
hg={
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes',hls),
    'alpha': hp.loguniform('alpha', np.log(alpha[0]), np.log(alpha[-1])),
    'learning_rate': hp.choice('learning_rate',lr),
    'learning_rate_init': hp.loguniform('learning_rate_init', np.log(lr_init[0]),np.log(lr_init[-1])),
    'random_state': hp.choice('random_state', rs),
    'solver': hp.choice('solver',solver)
}

# skopt paramgrid
bg = {
    'hidden_layer_sizes': Categorical(hls),
    'alpha': Real(alpha[0], alpha[-1], 'loguniform'),
    'learning_rate': Categorical(lr),
    'learning_rate_init': Real(lr_init[0],lr_init[-1], 'logunifrom'),
    'random_state': rs,
    'solver': Categorical(solver)
}

# base model parameters
base = {
    'hidden_layer_sizes':(n_features,),
    'alpha':0.001,
    'learning_rate': lr[0],
    'learning_rate_init': 0.001,
    'random_state':1,
    'solver' : solver[0]
}


# RUN COMPARISON

print('RUNNING COMPARISON')
#res = cmp_hpt_methods_double_cv(data, **mlpc)

hpt_objs = [
    HPT_OBJ('Baseline', base, run_baseline, {'cv':CV_SPLITS}),
    #        HPT_OBJ('Grid Search', pg, grid_search, {'cv':CV_SPLITS, 'refit':'loss'}),
        HPT_OBJ('Random Search', pg, random_search, {'n_iter': max_iter, 'cv':CV_SPLITS, 'refit':'loss'}),
    HPT_OBJ('Bayes Search', bg, baysian_search, {'n_iter':max_iter, 'cv':CV_SPLITS}),
    HPT_OBJ('Tree of Parzen Est.', hg, tpe_search, {'cv':CV_SPLITS, 'max_iter': max_iter}),
]

scoring = {
    'loss': make_scorer(log_loss, greater_is_better=True, needs_proba=True, labels=dsBunch.target),
    'acc': make_scorer(accuracy_score),
}
    #scoring = make_scorer(log_loss, greater_is_better=True, needs_proba=True, labels=sorted(np.unique(data[1])))
scoring =  make_scorer(accuracy_score)

mlpc ={
    'model': neural_network.MLPClassifier,
    'hpt_objs': hpt_objs,
    'score': scoring,
    'final_metric': accuracy_score,
    'iters' : [ i for i in range(st, end, step)],
    'name': '{}_{}'.format(name,max_iter),
    'random_state': 1,
    #    'cv': CV_SPLITS
}

res = cmp_hpt_methods(data, **mlpc)
