import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from time import time

from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint as sp_randint

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def test_random_tree_classifier():
    wine = pd.read_csv("winequality-red.csv")

    bins = (2, 6, 8)
    group_names = ['bad', 'good']
    wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

    label_quality = LabelEncoder()
    wine['quality'] = label_quality.fit_transform(wine['quality'])
    wine['quality'].value_counts()

    X = wine.drop('quality', axis = 1)
    y = wine['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)

    print(classification_report(y_test, pred_rfc))


    rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)

    print("The mean cross-validation score is ", rfc_eval.mean())
    # Maximum number

    random_grid = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}



    rf_grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv =3, n_jobs = -1, verbose=2)
    start = time()
    rf_grid.fit(X_train, y_train)
    print("GridSearchCV took ", (time()-start), " seconds.")
    report(rf_grid.cv_results_)

    rf_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=60, cv=3, verbose=2, n_jobs=-1)
    start = time()
    rf_random.fit(X_train, y_train)
    print("RandomizedSearchCV took ",(time()-start)," seconds.")
    report(rf_random.cv_results_)

    best_grid = rf_grid.best_estimator_
    best_random = rf_random.best_estimator_




def main():
    test_random_tree_classifier()

if __name__ == "__main__":
    main()