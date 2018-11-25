import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from hpt_cmp import HPT_METHOD

PLOT_DIR = 'plots'

#--------VISUALISE/SUMMARY FUNCTIONALITY------------
'''
transform results into pandas DataFrame
'''
def table(results, columns=[]):
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

def saveplot(name):
    path = './{}/{}'.format(PLOT_DIR, name)
    plt.savefig(path)



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
