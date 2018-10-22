import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

#https://www.youtube.com/watch?v=Gol_qOgRqfA
def test_cross_val_score():
    iris = load_iris()

    X = iris.data
    y = iris.target

    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
    #print(scores.mean())

    k_range = range(1, 31)
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    #print(k_scores)

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')



    param_grid = dict(n_neighbors=k_range)
    #print(param_grid)

    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

    grid.fit(X, y)

    #print(grid.cv_results_['params'][0])
    #print(grid.cv_results_['mean_test_score'][0])

    grid_mean_scores = grid.cv_results_['mean_test_score']
    #print(grid_mean_scores)

    #print(grid.best_score_)
    #print(grid.best_params_)
    #print(grid.best_estimator_)

    knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')

    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    param_grid = dict(n_neighbors = k_range, weights = weight_options)

    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy', return_train_score=False)
    grid.fit(X, y)

    grid.cv_results_

    print(grid.best_score_)
    print(grid.best_params_)

    knn.fit(X, y)

    print(knn.predict([[3,5,4,2]]))

    print(grid.predict([[3,5,4,2]]))
    #RANDOMIZED SEARCH CV

    param_dist = dict(n_neighbors = k_range, weights = weight_options)

    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, return_train_score=False)
    rand.fit(X,y)

    print(rand.best_score_)
    print(rand.best_params_)

    best_scores = []
    for _ in range(20):
        rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, return_train_score=False)
        rand.fit(X,y)
        best_scores.append(round(rand.best_score_, 3))

    print(best_scores)

    #Start with GridSearchCV and switch to RandomSearchCV when grid is taking longer than available time.
    #When using Rand, start with small n_iter, time how long it takes, then do the math for how large n_iter can be
    #without running over available time.

#https://www.youtube.com/watch?v=1nWFHa6K23w
#def test_logistic_regression():

    #cars = pd.read_csv("mtcars.csv")

    #cars.columns = ['car_names', "mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]
    #cars_data = cars.ix[:, (5, 11)].values
    #cars_data_names = ["drat", "carb"]
    #y = cars.ix[:,9].values
    #X = scale(cars_data)
    #LogReg = LogisticRegression()
    #LogReg.fit(X, y)
    #print(LogReg.score(X, y))
    #y_pred = LogReg.predict(X)
    #print(classification_report(y, y_pred))

#https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/
def test_logistic_regression_2():
    iris = load_iris()
    X = iris.data
    y = iris.target

    logistic = linear_model.LogisticRegression()

    penalty = ['l1', 'l2']
    C = np.logspace(0, 4, 10)
    hyperparameters = dict(C=C, penalty=penalty)

    clf = GridSearchCV(logistic, hyperparameters, cv=5,verbose=0)

    best_model = clf.fit(X,y)

    print("Best Penalty:", best_model.best_estimator_.get_params()['penalty'])
    print("Best C:", best_model.best_estimator_.get_params()['C'])

    print(best_model.predict(X))


#From youtube video (https://www.youtube.com/watch?v=aeWmdojEJf0)
def test_linear_regression():

    #Dataset from sklearn about housing prices
    boston = load_boston()

    #Dataframes
    df_x = pd.DataFrame(boston.data, columns= boston.feature_names)
    df_y = boston.target

    #Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)

    #Perform linear regression
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    #find
    a = reg.predict(x_test)

    #return mean square
    return np.mean((a - y_test)**2)

def main():

    #print(test_linear_regression())
    #print(test_logistic_regression_2())
    #test_cross_val_score()
    #test_multiple_params()

if __name__ == '__main__':
  main()