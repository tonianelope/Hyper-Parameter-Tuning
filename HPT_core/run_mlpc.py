import dataset_loader as ds
import model_loader as mdl
import seaborn as sns

from hpt_cmp import *

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, neural_network
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score ,make_scorer, log_loss, recall_score
from tqdm import tqdm, tnrange
from skopt.space import Real, Integer, Categorical

import warnings
warnings.filterwarnings('ignore')

MAX_ITER = 20
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
        HPT_OBJ('Grid Search', pg, grid_search, {'cv':CV_SPLITS, 'refit':'loss'}),
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
    'name': NAME
}

# RUN COMPARISON
print('RUNNING COMPARISON')
res = cmp_hpt_methods_double_cv(data, **mlpc)

# RUN PLOTS

sum_res = []
for r in res:
    t = np.array(r[INNER_RES][0]['mean_fit_time']).mean()
    score_label = 'mean_test_loss'
    if r[HPT_METHOD] == 'Bayes Search':
        score_label = 'mean_test_score'
    sum_res.append((r[HPT_METHOD], r[MEAN+CV_TIME],len(r[INNER_RES][0]['params']), r[MEAN+TEST_ACC], r[BEST_PARAMS], np.array(r[INNER_RES][0][score_label]).mean() ))

                   
df = pd.DataFrame(sum_res, columns=[HPT_METHOD, 'TIME', PARAMS_SAMPLED, TEST_ACC, BEST_PARAMS, 'SCORE'])

print('RESULTS:\n')
print(df)

fig, ax=plt.subplots()
sns.barplot(x='TIME', y=HPT_METHOD,data =df,ax=ax)
ax.set_xlabel('Cross-validation time in seconds')
ax.set_yticklabels(['No HPT','Grid', 'Random', 'Bayes', 'TPE'])
fig.tight_layout()
plt.savefig('./plots/{}-TIME'.format(NAME))

plt.figure()
# plot accuracy comparison
fig, ax =plt.subplots()
sns.barplot(y=TEST_ACC, x=HPT_METHOD, ax=ax,data =df)
ax.set_ylabel('Validation Accuracy')
ax.set_xticklabels(['No HPT','Grid', 'Random', 'Bayes', 'TPE'])
fig.tight_layout()
plt.savefig('./plots/{}-FULLACC'.format(NAME))

fig, ax =plt.subplots()
ax.set(xlim=(0.5, 1.0))
sns.barplot(y=TEST_ACC, x=HPT_METHOD, ax=ax,data =df)
ax.set_ylabel('Validation Accuracy')
ax.set_xticklabels(['No HPT','Grid', 'Random', 'Bayes', 'TPE'])
fig.tight_layout()
plt.savefig('./plots/{}-PARTACC'.format(NAME))

