#!/usr/bin/env python3

from sklearn.ensemble import AdaBoostClassifier
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Run experiments.
    ex1 = experiment(X_train, X_test, y_train, y_test,
                     None, 50, 1, 'SAMME.R', None)
    print(f"accuracy ex1: {ex1}")
