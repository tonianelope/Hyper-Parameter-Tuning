from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

data = load_breast_cancer()
C = [0.1,1,10,100]
gamma = [0.001,0.01,0.1,1]
kernels=['linear','rbf']
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
best_score=0.0
c_score=0.0
g_score=0.0
kernel_type=""
for k in kernels:
    for c in C:
        for g in gamma:
            clf = svm.SVC(kernel=k, C=c, gamma=g)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if metrics.accuracy_score(y_test,y_pred)>best_score:
                best_score = metrics.accuracy_score(y_test,y_pred)
                c_score=c
                g_score=g
                kernel_type=k
print("best score = ", best_score, " when C was ", c_score, " and when gamma was ", g_score, " and kernel was ",kernel_type )

#clf = svm.SVC(kernel='linear')
#clf = GridSearchCV(svc,parameters,cv=5)
#clf.fit(X_train, y_train)
#print(clf.best_params_);
#y_pred = clf.predict(X_test);

#print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
