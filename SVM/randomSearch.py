from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
parameters = {'kernel': ('linear', 'rbf'), 'C':[0.001,0.01,0.1,1,10],'gamma':[0.001,0.01,0.1,1]}
svc = svm.SVC()
clf = RandomizedSearchCV(svc,parameters,n_iter=40)
clf.fit(X_train,y_train)
print(clf.best_params_)
y_pred = clf.predict(X_test);
print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
