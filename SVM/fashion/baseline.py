import mnist_reader
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import timeit

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train[:10000]
y_train = y_train[:10000]

start = timeit.default_timer()
kernel_name = 'poly'
clf = svm.SVC(kernel=kernel_name)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
stop = timeit.default_timer()


print("*****FASHION BASELINE DATA with kernel=", kernel_name, "*****")
print("accuracy: ", metrics.accuracy_score(y_test,y_pred))
print("Time: ", stop - start)
