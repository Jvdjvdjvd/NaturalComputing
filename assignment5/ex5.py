#!/usr/bin/env python3

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics


def experiment(X_train, X_test, y_train, y_test,
               base_estimator, n_estimators, learning_rate, algorithm,
               random_state):
    ada = AdaBoostClassifier(base_estimator=base_estimator,
                             n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             algorithm=algorithm,
                             random_state=random_state)
    classifier = ada.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    # Grab the initial dataset.
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target

    # Split the set into train and test subsets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Defaults.
    base_estimator = None
    n_estimators = 50
    learning_rate = 1
    algorithm = "SAMME.R"
    random_state = None

    # Run experiments.
    ex1 = experiment(X_train, X_test, y_train, y_test,
                     base_estimator, n_estimators, learning_rate, algorithm,
                     random_state)
    ex2 = experiment(X_train, X_test, y_train, y_test,
                     base_estimator, n_estimators, learning_rate, "SAMME",
                     random_state)
    ex3 = experiment(X_train, X_test, y_train, y_test,
                     base_estimator, n_estimators, 0.8, algorithm,
                     random_state)
    ex4 = experiment(X_train, X_test, y_train, y_test,
                     base_estimator, 5, learning_rate, algorithm,
                     random_state)
    ex5 = experiment(X_train, X_test, y_train, y_test,
                     SVC(kernel="linear"), n_estimators, learning_rate, "SAMME",
                     random_state)
    print(f"accuracy ex1, defaults: {ex1}")
    print(f"accuracy ex2, defaults with SAMME: {ex2}")
    print(f"accuracy ex3, defaults with learning rate 0.8: {ex3}")
    print(f"accuracy ex4, defaults with 10 classifiers: {ex4}")
    print(f"accuracy ex5, defaults with svc classifier and SAMME: {ex5}")
    print("---")
