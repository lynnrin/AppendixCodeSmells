import pandas as pd
import numpy as np
import random

from cleanlab.noise_generation import generate_noise_matrix_from_trace, generate_noisy_labels


def input_dataset(first_project, projects, test_path):
    index_name = pd.read_csv(first_project).columns.values[0].split('.')

    append_df = pd.read_csv(first_project, sep=',', skiprows=1,
                            header=None, names=index_name, quoting=3).drop(columns='name').replace('(.*)"(.*)', r'\1\2',
                                                                                                   regex=True)
    for project_path in projects:
        df = pd.read_csv(project_path, sep=',', skiprows=1,
                         header=None, names=index_name, quoting=3).drop(columns='name').replace('(.*)"(.*)', r'\1\2',
                                                                                                regex=True)
        append_df = pd.concat([append_df, df], join='outer')

    y_train = append_df[['isSmelly']].replace('true', 1).replace('false', 0).values.reshape(-1)
    X_train = append_df.drop(columns='isSmelly').values

    index_name = pd.read_csv(test_path).columns.values[0].split('.')
    df2 = pd.read_csv(test_path, sep=',', skiprows=1,
                      header=None, names=index_name, quoting=3).drop(columns='name').replace('(.*)"(.*)', r'\1\2',
                                                                                             regex=True)

    y_test = df2[['isSmelly']].replace('true', 1).replace('false', 0).values.reshape(-1)
    X_test = df2.drop(columns='isSmelly').values

    return X_train, y_train, X_test, y_test


def add_outlier(y_train):
    seed = random.randint(0, 1000)
    a, py = np.unique(y_train, return_counts=True)
    noise_matrix = generate_noise_matrix_from_trace(2, 1.95, min_trace_prob=0.15, py=py, seed=seed)
    np.random.seed(seed)
    y_train_corrupted = generate_noisy_labels(y_train, noise_matrix)
    y_train_is_error = y_train_corrupted != y_train
    n = y_train_is_error.sum()
    return y_train_corrupted, int(n / len(y_train) * 100)

