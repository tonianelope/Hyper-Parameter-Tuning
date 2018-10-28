from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import timeit

start = timeit.default_timer()
print("*****BREAST CANCER GRID SEARCH*****")
data = load_breast_cancer()
C = [0.001,0.01,0.1,1,10,100]
gamma = [0.001,0.01,0.1,1]
kernels = ['linear','rbf']
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
parameters = {'kernel': ('linear', 'rbf'), 'C':[0.001,0.01,0.1,1,10],'gamma':[0.001,0.01,0.1,1]}
svc = svm.SVC()
clf = GridSearchCV(svc,parameters)
clf.fit(X_train,y_train)
print(clf.best_params_)
y_pred = clf.predict(X_test);
stop = timeit.default_timer()

print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Time: ", stop - start)
