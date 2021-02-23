import pandas as pd
import numpy as np

import random
from multiprocessing import cpu_count

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from cleanlab.classification import LearningWithNoisyLabels


def model_select(X_train, y_train, X_test, alg, switch):
    if switch == 'baseline':
        pred = __model_build_baseline(X_train, y_train, X_test, alg)
    elif switch == 'noisy':
        pred = __model_build_noisy(X_train, y_train, X_test, alg, random.randint(0, 1000))
    elif switch == 'pseudo':
        pred = __model_build_noisy_pseudo(X_train, y_train, X_test, alg, random.randint(0, 1000))
    return pred


def __model_build_baseline(X_train, y_train, X_test, alg):
    clf = GaussianNB()
    if alg == 'Logistic':
        clf = LogisticRegression(multi_class='auto')

    clf.fit(X_train, y_train)

    return clf.predict(X_test)


def __model_build_noisy(X_train, y_train, X_test, alg, seed):
    model = GaussianNB()
    if alg == 'Logistic':
        model = LogisticRegression(multi_class='auto')
    clf = LearningWithNoisyLabels(clf=model, seed=seed, n_jobs=cpu_count())
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def __model_build_noisy_pseudo(X_train, y_train, X_test, alg, seed):
    model = GaussianNB()
    if alg == 'Logistic':
        model = LogisticRegression(multi_class='auto')

    clf = LearningWithNoisyLabels(clf=model, seed=seed, n_jobs=cpu_count())
    clf.fit(X_train, y_train)

    # Pseudo-labelling
    X_with_noise = X_train[clf.noise_mask]
    y_train_pseudo = y_train.copy()
    y_train_pseudo[clf.noise_mask] = clf.predict(X_with_noise)
    y_test_pseudo = clf.predict(X_test)
    y_pseudo = np.hstack([y_train_pseudo, y_test_pseudo])
    X_for_pseudo = np.vstack([X_train, X_test])
    model.fit(X_for_pseudo, y_pseudo)
    return model.predict(X_test)


def calc_result(y_test, pred):
    SmP = precision_score(y_test, pred, average="binary", pos_label=1)
    SmR = recall_score(y_test, pred, average="binary", pos_label=1)
    SmF = f1_score(y_test, pred, average="binary", pos_label=1)

    NSmP = precision_score(y_test, pred, average="binary", pos_label=1)
    NSmR = recall_score(y_test, pred, average="binary", pos_label=1)
    NSmF = f1_score(y_test, pred, average="binary", pos_label=1)

    MCC = matthews_corrcoef(y_test, pred)

    col = ['Smell Precision', 'Smell Recall', 'Smell F1', 'NonSmell Precision', 'NonSmell Recall', 'NonSmell F1', 'MCC']
    return pd.DataFrame([[SmP, SmR, SmF, NSmP, NSmR, NSmF, MCC]],
                        columns=col, index=None)


def count_diff(pred, y_test):
    C = pred * 2 + y_test
    print(np.unique(C).size)
    a, py = np.unique(C, return_counts=True)
    print(a)
    print(py)
    return C
