from os import listdir, path

import seaborn as sns

import dataset_loader as ds
from hpt_cmp import *
from plots import *

PLOT_DIR = './plots'
DATA_DIR = './output'

name = 'test-plots'

onlyfiles = [f for f in listdir(DATA_DIR) if path.isfile(path.join(DATA_DIR, f))]

results = []

dsBunch = ds.load_mnist_back()
n_features = dsBunch.data.shape[1]
# DEFINE PARAM GRIDS
d_features = n_features*0.5
hls = [(d_features,)*4, (n_features,)*4, (d_features,)*2, (n_features,)*2, (d_features,), (n_features,),]
alpha = [0.0001, 0.001, 0.01, 0.1]
lr = ['adaptive','constant','invscaling']
lr_init = [0.00001, 0.0001, 0.001, 0.01, 0.1]
rs = [1]

for f in onlyfiles:
    with open(path.join(DATA_DIR,f)) as json_data:
        r =  json.load(json_data)
        results.append(r)

iterations = []
for it, _ in enumerate(results[1][ITER_DATA]):
    iterations.append(it)

print('ITER')
print(iterations)

# plot test vs train score evolution
for r in results:
    print(r[HPT_METHOD])
    print()
    plt.figure()
    x = [i for i in range(len(r[INNER_RES]['mean_test_score']))]
    sns.lineplot(y=r[INNER_RES]['mean_test_score'], x=x, label='test '+r[HPT_METHOD])
    try:
        x = [i for i in range(len(r[INNER_RES]['mean_train_score']))]
        sns.lineplot(y=r[INNER_RES]['mean_train_score'], x=x, label='train')
    except:
        pass
    plt.show()
    saveplot('{}_x.png'.format(r[HPT_METHOD]))

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

    #short_hptm = ['No HPT','Grid', 'Random', 'Bayes', 'TPE']
    short_hptm = ['No HPT','Random', 'Bayes', 'TPE']
    #MEAN= ''

    barplot(MEAN+CV_TIME, HPT_METHOD, df, textval=MEAN+CV_TIME, xlabel=MEAN+CV_TIME, ytick=short_hptm)
    saveplot('{}-{}-Time.png'.format(name, it))

    barplot(HPT_METHOD, TEST_ACC, df, textval=TEST_ACC, ylabel=TEST_ACC, xtick=short_hptm)
    saveplot('{}-{}-Accuracy.png'.format(name, it))

    barplot(HPT_METHOD, TEST_ERR, df, textval=TEST_ERR, ylabel=TEST_ERR, xtick=short_hptm)
    saveplot('{}-{}-Error.png'.format(name, it))


    params = ['alpha', 'learning_rate_init']
    scatterplot_param_distribution(res, params, 'mean_test_score')
    scatterplot_param_distribution(res, ['hidden_layer_sizes'], 'mean_test_score', hls)
    scatterplot_param_distribution(res, ['learning_rate'], 'mean_test_score', lr)


for i in iterations:
    plot_res(results, i)
