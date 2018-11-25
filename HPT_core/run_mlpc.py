import warnings

import seaborn as sns
from sklearn import (discriminant_analysis, ensemble, gaussian_process,
                     linear_model, naive_bayes, neighbors, neural_network, svm,
                     tree)
from sklearn.metrics import (accuracy_score, f1_score, log_loss, make_scorer,
                             recall_score, roc_auc_score)
from skopt.space import Categorical, Integer, Real
from tqdm import tnrange, tqdm

import dataset_loader as ds
import model_loader as mdl
from hpt_cmp import *
from hpt_methods import *
from plots import *

warnings.filterwarnings('ignore')

MAX_ITER = 5
CV_SPLITS = 2
NAME = "RUN-1-MPLC"

# LOAD DATA
dsBunch = ds.load('iris')
data = (dsBunch.data, dsBunch.target)
print('DATA:')
n_features = dsBunch.data.shape[1]
shp = dsBunch.data.shape
print('n_features: {}\nshape: {}\n'.format(n_features, shp))

# DEFINE PARAM GRIDS
d_features = n_features*4
hls = [(d_features,)*5, (n_features,)*5, (d_features,)*2, (n_features,)*2, (d_features,), (n_features),]
alpha = [0.0001, 0.001, 0.01, 0.1]
lr = ['adaptive'] #,'constant','invscaling']
lr_init = [0.00001, 0.0001, 0.001, 0.01, 0.1]
rs = [1]

# sklean paramgrid
pg = {
    'hidden_layer_sizes': hls,
    'alpha': alpha,
    'learning_rate': lr,
    'learning_rate_init': lr_init,
    'random_state': rs
}

# hyperopt paramgird
hg={
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes',hls),
    'alpha': hp.loguniform('alpha', np.log(alpha[0]), np.log(alpha[-1])),
    'learning_rate': hp.choice('learning_rate',lr),
    'learning_rate_init': hp.loguniform('learning_rate_init', np.log(lr_init[0]),np.log(lr_init[-1])),
    'random_state': hp.choice('random_state', rs)
}

# skopt paramgrid 
bg = {
    'hidden_layer_sizes': Categorical(hls),
    'alpha': Real(alpha[0], alpha[-1], 'loguniform'),
    'learning_rate': Categorical(lr),
    'learning_rate_init': Real(lr_init[0],lr_init[-1], 'logunifrom'),
    'random_state': rs
}

# base model parameters
base = {
    'hidden_layer_sizes':(n_features,), 
    'alpha':0.001,
    'learning_rate': lr[0],
    'learning_rate_init': 0.001,
    'random_state':1}


# DEFINE MLPClassifier
hpt_objs = [
        HPT_OBJ('Baseline', base, run_baseline, {'cv':CV_SPLITS}),
#        HPT_OBJ('Grid Search', pg, grid_search, {'cv':CV_SPLITS, 'refit':'loss'}),
        HPT_OBJ('Random Search', pg, random_search, {'n_iter': MAX_ITER, 'cv':CV_SPLITS, 'refit':'loss'}),
        HPT_OBJ('Bayes Search', bg, baysian_search, {'n_iter':MAX_ITER, 'cv':CV_SPLITS}),
        HPT_OBJ('Tree of Parzen Est.', hg, tpe_search, {'cv':CV_SPLITS, 'max_iter': MAX_ITER}),
]


scoring = {
    'loss': make_scorer(log_loss, greater_is_better=True, needs_proba=True, labels=dsBunch.target),
    'acc': make_scorer(accuracy_score),
}
#scoring = make_scorer(log_loss, greater_is_better=True, needs_proba=True, labels=sorted(np.unique(data[1])))
#scoring =  make_scorer(accuracy_score)

mlpc ={
    'model': neural_network.MLPClassifier,
    'hpt_objs': hpt_objs,
    'score': scoring,
    'final_metric': accuracy_score,
    'name': NAME,
    'cv': CV_SPLITS
}

# RUN COMPARISON
print('RUNNING COMPARISON')
res = cmp_hpt_methods_double_cv(data, **mlpc)
#res = cmp_hpt_methods(data, **mplc)

# RUN PLOTS

sum_res = []
cols = [HPT_METHOD, MEAN+CV_TIME, PARAMS_SAMPLED, MEAN+TEST_ACC, MEAN+TEST_ERR, STD_TEST_SCR, BEST_PARAMS]
for r in res:
    row = []
    for c in cols:
        if isinstance(r[c], list):
            row.append(r[c][0])
        else:
            row.append(r[c])
    sum_res.append(row)

df = pd.DataFrame(sum_res, columns=cols)
df.to_csv('./plots/'+NAME+'-data.csv')

print('RESULTS:\n')
print(df)

#short_hptm = ['No HPT','Grid', 'Random', 'Bayes', 'TPE']
short_hptm = ['No HPT','Random', 'Bayes', 'TPE']

barplot(MEAN+CV_TIME, HPT_METHOD, df, textval=MEAN+CV_TIME, xlabel=MEAN+CV_TIME, ytick=short_hptm)
saveplot(NAME+'-Time')

barplot(HPT_METHOD, MEAN+TEST_ACC, df, textval=MEAN+TEST_ACC, ylabel=MEAN+TEST_ACC, xtick=short_hptm)
saveplot(NAME+'-Accuracy')

barplot(HPT_METHOD, MEAN+TEST_ERR, df, textval=MEAN+TEST_ERR, ylabel=MEAN+TEST_ERR, xtick=short_hptm)
saveplot(NAME+'-Error')
