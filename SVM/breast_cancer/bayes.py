from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')
    
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=142)

opt = BayesSearchCV(
    SVC(),
    {
        'C': Real(0.001, 10, prior='log-uniform'),
        'gamma': Real(0.001, 1, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf']),
    },
)
opt.fit(X_train, y_train)

print(opt.score(X_test, y_test))
print(opt.best_params_)
