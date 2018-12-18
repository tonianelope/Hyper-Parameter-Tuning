from os import listdir, path

import numpy as np
import seaborn as sns

import dataset_loader as ds
from hpt_cmp import *
from plots import *


def plot_params_vs_score(all_params, cor_df, lables, param_labels, scaled=True):
    fig, axs = plt.subplots(1,3, figsize=(24,6))

    for i, hyper in enumerate(all_params[1:]):
        if scaled : axs[i].set_ylim([0,1])
        sns.regplot(hyper, 'mean_test_score', data=cor_df, ax=axs[i])
        #axs[i].scatter(best_random_hyp[hyper], best_random_hyp['mean_test_score'], marker = '*', s = 200, c = 'k')
        axs[i].set(xlabel = '{}'.format(param_labels[i]), ylabel = 'Accuracy', title = 'Accuracy vs {}'.format(param_labels[i]));
        if not hyper=='param_alpha':
            axs[i].set_xticks([i for i in range(len(lables[i]))])
            axs[i].set_xticklabels(lables[i], rotation=45)
        
    fig.tight_layout()
    if scaled:
        saveplot('{}-hyper-scaled.png'.format(name))
    else:
        saveplot('{}-hyper.png'.format(name))

# RUN PLOTS
def plot_res(res, it):
    sum_res = []
    MEAN= 'Mean '
    cols = [HPT_METHOD, MEAN+CV_TIME, PARAMS_SAMPLED, TEST_ACC, TEST_ERR, STD_TEST_SCR, BEST_PARAMS]

    for r in res:
        row = []
        row.append(r[HPT_METHOD])
        row.append(r[CV_TIME])
        for c in cols[2:]:
            if isinstance(r[ITER_DATA][it][c], list):
                row.append(r[ITER_DATA][it][c][0])
            else:
                row.append(r[ITER_DATA][it][c])
        sum_res.append(row)

    df = pd.DataFrame(sum_res, columns=cols)
    df.to_csv('./plots/{}-{}-data.csv'.format(name, it))

    print('RESULTS:\n')
    print(df)

    short_hptm = ['No HPT','Grid', 'Random', 'Bayes', 'TPE']
    #short_hptm = ['No HPT', 'Random', 'Bayes', 'TPE']
    #MEAN= ''

    barplot(MEAN+CV_TIME, HPT_METHOD, df, textval=MEAN+CV_TIME, xlabel=MEAN+CV_TIME, ytick=short_hptm)
    saveplot('{}-{}-Time.png'.format(name, it))

    barplot(HPT_METHOD, TEST_ACC, df, textval=TEST_ACC, ylabel=TEST_ACC, xtick=short_hptm)
    saveplot('{}-{}-Accuracy.png'.format(name, it))

    barplot(HPT_METHOD, TEST_ERR, df, textval=TEST_ERR, ylabel=TEST_ERR, xtick=short_hptm)
    saveplot('{}-{}-Error.png'.format(name, it))


    params = ['alpha']
    scatterplot_param_distribution(res, params, 'mean_test_score')
    scatterplot_param_distribution(res, ['hidden_layer_sizes'], 'mean_test_score', hls)
   # scatterplot_param_distribution(res, ['alpha'], 'mean_test_score', alpha)
    try:
        scatterplot_param_distribution(res, ['activation'], 'mean_test_score', activations)
    except e:
        pass



PLOT_DIR = './plots'
DATA_DIR = './outt'

onlyfiles = [f for f in listdir(DATA_DIR) if path.isfile(path.join(DATA_DIR, f))]

results = []

#dsBunch = ds.load_mnist_back()
n_features = 784
# DEFINE PARAM GRIDS
# hls = [[256,128,100], [256, 100], [127,100],[128,100], [256,], [128,], [100,],]
hls = [[256,128,100], [256, 128],[256,100], [128,100],[100,10],[256], [128], [100,],[10,10],[10,]]
alpha = [0.0001, 0.001, 0.01, 0.1]
#solver = ['adam', 'sgd']
activations = ['relu', 'tanh', 'logistic']
rs = [1]

#
files = ['-Baseline.json', '-Grid Search.json', '-Bayes Search.json', '-Random Search.json', '-Tree of Parzen Est..json']
prefix = 'fa-a_50'
name = prefix

for i, s in enumerate(files):
    files[i] = prefix+s

for f in files:
    with open(path.join(DATA_DIR,f)) as json_data:
        r =  json.load(json_data)
        results.append(r)

acc_over_it = {
}
iterations = []
for it, k in enumerate(results[1][ITER_DATA]):
    iterations.append(it)


for r in results:
    acc_over_it[r[HPT_METHOD]] = {
        TEST_ACC : [],
        PARAMS_SAMPLED: []
    }
    for k in r[ITER_DATA]:
        acc_over_it[r[HPT_METHOD]][TEST_ACC].append(k[TEST_ACC])
        acc_over_it[r[HPT_METHOD]][PARAMS_SAMPLED].append(k[PARAMS_SAMPLED])

    a_df = pd.DataFrame(acc_over_it[r[HPT_METHOD]])
    sns.lineplot(x=PARAMS_SAMPLED, y=TEST_ACC, data=a_df, label=r[HPT_METHOD])
saveplot('{}-iter.png'.format(name))


print('ITER')
print(iterations)

# plot test vs train score evolution
for r in results:
    plt.figure()
    x = [i for i in range(len(r[INNER_RES]['mean_test_score']))]
    sns.lineplot(y=r[INNER_RES]['mean_test_score'], x=x, label='test '+r[HPT_METHOD])
    try:
        x = [i for i in range(len(r[INNER_RES]['mean_train_score']))]
        sns.lineplot(y=r[INNER_RES]['mean_train_score'], x=x, label='train')
    except:
        pass
    #plt.show()
saveplot('{}_{}_x.png'.format(name, r[HPT_METHOD]))

# PLOT CORROLATION
all_params = [ 'mean_test_score','param_alpha', 'param_hidden_layer_sizes', 'param_activation']
param_labels = [ 'accuracy','alpha', 'hidden layers', 'activation']
data = {}
for i in all_params:
    data[i] = []

for r in results[1:]:
    for p in all_params:
        data[p]+=(r[INNER_RES][p])

cor_df = pd.DataFrame(data)
data_df = pd.DataFrame(data)
for i in data_df:
    if isinstance(data_df[i][0], list):
        strs = []
        for e in data_df[i]:
            print(e)
            strs.append(str(e))
        print(strs)
        data_df[i] = strs
    data_df[i] = data_df[i].astype('category').cat.as_ordered()

cor_df['param_hidden_layer_sizes'] = cor_df['param_hidden_layer_sizes'].map(lambda x: len(hls)-hls.index(x))
# cor_df['param_hidden_layer_sizes'] = cor_df['param_hidden_layer_sizes'].astype('category').cat.codes
cor_df['param_activation'] = cor_df['param_activation'].map(lambda x: activations.index(x))#cor_df['param_activation'].astype('category').cat.codes
#cor_df['param_alpha'] = cor_df['param_alpha'].astype('category').cat.codes

cor = cor_df.corr()

fig, ax = plt.subplots()
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(cor, mask= mask, xticklabels=param_labels, yticklabels=param_labels, annot=True, cmap=cmap)#plt.cm.Blues)
fig.tight_layout()
saveplot('{}-corr.png'.format(name))


best_random_hyp = {}
indx = np.argmax(data['mean_test_score'])
for i in all_params:
    best_random_hyp[i] = data_df[i][indx]

plot_params_vs_score(all_params, cor_df, [[], hls, activations], param_labels)
plot_params_vs_score(all_params, cor_df, [[], hls, activations], param_labels,scaled=False)

for i in iterations:
    plot_res(results, i)
