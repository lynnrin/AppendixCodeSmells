import parameter
from dataset_operation import input_dataset, add_outlier
from learning import model_select, calc_result, count_diff

import traceback
import re
import os
import requests

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')


def operate():
    for sm in parameter.smellNames:
        print('==========================')
        print('smell name: {0}'.format(sm))

        for p in parameter.projectNames:
            print('------------------')
            print('\n\ntarget: {0}\tmodel: {1}\n'.format(p[0], p[-1]))
            X_train, y_train, X_test, y_test = input_dataset(parameter.return_path(sm, p[0]),
                                                             parameter.return_paths(sm, p[1:-1]),
                                                             parameter.return_path(sm, p[-1]))

            bayes_base, log_base, pred_Bayes, pred_Log, flag = \
                __baseline_learning_operate(X_train, y_train, X_test, y_test)

            bayes_noi, log_noi, bayes_pseudo, log_pseudo = __noisy_learning_operate(X_train, y_train, X_test, y_test)

            bayes_add, log_add, bayes_add_noi, log_add_noi, bayes_add_pseudo, log_add_pseudo = \
                __add_outlier_learning_operate(X_train, y_train, X_test, y_test)

            m = re.findall('[a-z]+', p[-1])[0]
            if flag == 0:
                write_not_learning(sm, m)

            # save result
            create_dir(sm, m)
            bayes_base.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'Bayes_baseline'))
            write_count_pred(pred_Bayes, parameter.return_path(sm, p[-1]), y_test, sm, m, 'Bayes')
            log_base.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'Logistic_baseline'))
            write_count_pred(pred_Bayes, parameter.return_path(sm, p[-1]), y_test, sm, m, 'Logistic')

            bayes_noi.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'Bayes_noisy'))
            log_noi.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'Logistic_noisy'))
            bayes_pseudo.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'Bayes_pseudo'))
            log_pseudo.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'Logistic_pseudo'))
            bayes_add.to_csv('/Users/lynnrin/noise-learninng/{0}/{1}/{2}.csv'.format(sm, m, 'add_out_bayes_baseline'))
            log_add.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'add_out_logistic_baseline'))
            bayes_add_noi.to_csv('/Users/lynnrin/noise-learninng/{0}/{1}/{2}.csv'.format(sm, m, 'add_out_bayes_noisy'))
            log_add_noi.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'add_out_logistic_noisy'))
            bayes_add_pseudo.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'add_out_bayes_pseudo'))
            log_add_pseudo.to_csv('./{0}/{1}/{2}.csv'.format(sm, m, 'add_out_logistic_pseudo'))


def __baseline_learning_operate(X_train, y_train, X_test, y_test):
    # initial data
    result_Bayes_baseline = result_Logistic_baseline = \
        pd.DataFrame(columns=['Smell Precision', 'Smell Recall', 'Smell F1',
                              'NonSmell Precision', 'NonSmell Recall', 'NonSmell F1',
                              'MCC'])

    pred_Bayes_baseline = pred_Logistic_baseline = np.zeros(y_test.size)
    flag = 1

    # cannot learning
    if np.unique(y_train).size != 2:
        return result_Bayes_baseline, result_Logistic_baseline, np.zeros(y_test.size), np.zeros(y_test.size), 0

    try:
        pred_Bayes_baseline = model_select(X_train, y_train, X_test, 'Bayes', 'baseline')
        result_Bayes_baseline = calc_result(y_test, pred_Bayes_baseline)

        pred_Logistic_baseline = model_select(X_train, y_train, X_test, 'Logistic', 'baseline')
        result_Logistic_baseline = calc_result(y_test, pred_Logistic_baseline)

    except Exception as e:
        print(traceback.format_exc())
        print("error could't build ML-model")
        flag = 0

    finally:
        return result_Bayes_baseline, result_Logistic_baseline, pred_Bayes_baseline, pred_Logistic_baseline, flag


def __noisy_learning_operate(X_train, y_train, X_test, y_test):
    # initial data
    result_Bayes = result_Logistic = result_Bayes_pseudo = result_Logistic_pseudo = \
        pd.DataFrame(columns=['Smell Precision', 'Smell Recall', 'Smell F1',
                              'NonSmell Precision', 'NonSmell Recall', 'NonSmell F1',
                              'MCC'])

    # cannot learning
    if np.unique(y_train).size != 2:
        return result_Bayes.copy(), result_Logistic.copy(), result_Bayes.copy(), result_Logistic.copy()

    try:
        pass
        # for i in range(300):
        #     pred_Bayes = model_select(X_train, y_train, X_test, 'Bayes', 'noisy')
        #     result_Bayes = pd.concat([result_Bayes, calc_result(y_test, pred_Bayes)], join='inner')
        #
        #     pred_Logistic = model_select(X_train, y_train, X_test, 'Logistic', 'noisy')
        #     result_Logistic = pd.concat([result_Logistic, calc_result(y_test, pred_Logistic)], join='inner')
        #
        #     pred_Bayes_pseudo = model_select(X_train, y_train, X_test, 'Bayes', 'pseudo')
        #     result_Bayes_pseudo = pd.concat([result_Bayes_pseudo, calc_result(y_test, pred_Bayes_pseudo)], join='inner')
        #
        #     pred_Logistic_pseudo = model_select(X_train, y_train, X_test, 'Logistic', 'pseudo')
        #     result_Logistic_pseudo = pd.concat([result_Logistic_pseudo, calc_result(y_test, pred_Logistic_pseudo)],
        #                                        join='inner')

    except Exception as e:
        print(traceback.format_exc())
        print("error could't build ML-model")

    finally:
        return result_Bayes, result_Logistic, result_Bayes_pseudo, result_Logistic_pseudo


def __add_outlier_learning_operate(X_train, y_train, X_test, y_test) -> object:
    # initial data
    result_Bayes_noise = result_Logistic_noise = result_Bayes = result_Logistic = result_Bayes_pseudo = result_Logistic_pseudo = \
        pd.DataFrame(columns=['Smell Precision', 'Smell Recall', 'Smell F1',
                              'NonSmell Precision', 'NonSmell Recall', 'NonSmell F1',
                              'MCC'])

    # cannot learning
    if np.unique(y_train).size != 2:
        return result_Bayes, result_Logistic, result_Bayes_noise, result_Logistic_noise, result_Bayes, result_Logistic
        # return result_Bayes_pseudo
    try:
        error_rate_list = []

        for i in range(300):
            # add outlier
            y_train_add, error_rate = add_outlier(y_train)
            error_rate_list.append(error_rate)

            pred_B = model_select(X_train, y_train_add, X_test, 'Bayes', 'baseline')
            result_Bayes = pd.concat([result_Bayes, calc_result(y_test, pred_B)], join='inner')

            pred_Bayes = model_select(X_train, y_train_add, X_test, 'Bayes', 'noisy')
            result_Bayes_noise = pd.concat([result_Bayes_noise, calc_result(y_test, pred_Bayes)], join='inner')

            pred_BP = model_select(X_train, y_train_add, X_test, 'Bayes', 'pseudo')
            result_Bayes_pseudo = pd.concat([result_Bayes_pseudo, calc_result(y_test, pred_BP)], join='inner')

            pred_L = model_select(X_train, y_train_add, X_test, 'Logistic', 'baseline')
            result_Logistic = pd.concat([result_Logistic, calc_result(y_test, pred_L)], join='inner')

            pred_Logistic = model_select(X_train, y_train_add, X_test, 'Logistic', 'noisy')
            result_Logistic_noise = pd.concat([result_Logistic_noise, calc_result(y_test, pred_Logistic)], join='inner')

            pred_LP = model_select(X_train, y_train_add, X_test, 'Logistic', 'pseudo')
            result_Logistic_pseudo = pd.concat([result_Logistic_pseudo, calc_result(y_test, pred_LP)], join='inner')

        print('average error rate is {0}'.format(sum(error_rate_list) / len(error_rate_list)))

    except Exception as e:
        print(traceback.format_exc())
        print("error could't build ML-model")

    finally:
        return result_Bayes, result_Logistic, result_Bayes_noise, result_Logistic_noise, result_Bayes_pseudo, result_Logistic_pseudo


def write_count_pred(pred, test_path, y_test, smell_name, project_name, name):
    count = count_diff(pred, y_test)
    create_dir('count', smell_name)

    index_name = pd.read_csv(test_path).columns.values[0].split('.')
    df = pd.read_csv(test_path, sep=',', skiprows=1,
                     header=None, names=index_name, quoting=3).replace('(.*)"(.*)', r'\1\2', regex=True)

    with open('./count/{0}/{1}.txt'.format(str(smell_name), str(re.findall('[a-z]+', project_name)[0])), mode='w') as f:
        label, num = np.unique(count, return_counts=True)
        s0 = '{0}-{1}\n'.format(str(smell_name), str(project_name))
        s1 = 'label: {0}\n'.format(str(label))
        s2 = 'num: {0}\n'.format(str(num))
        f.write(s0 + s1 + s2)
    pred_df = df.assign(pred=count)
    create_dir('pred', name)
    pred_df.to_csv('./pred/{0}/{1}_{2}.csv'.format(name, str(smell_name), str(project_name)))


def write_not_learning(smell_name, project_name):
    file_name = './not_leaning.txt'
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            f.write('unique key is not 2\n\n')
    with open(file_name, mode='a') as f:
        f.write('smell: {0}, project: {1}\n'.format(smell_name, project_name))


def create_dir(a, b):
    if not os.path.isdir('./{0}/'.format(str(a))):
        os.mkdir('./{0}'.format(str(a)))
    if not os.path.isdir('./{0}/{1}'.format(str(a), str(b))):
        os.mkdir('./{0}/{1}/'.format(str(a), str(b)))


def send_line(message: str):
    line_token = 'nU7wXOhxtTSvpZI0HENbOJEIPAGXh4rp1oJufBt98GQ'
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    header = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=header)


if __name__ == '__main__':
    send_line('start')
    try:
        operate()
    except Exception as e:
        send_line('die')
    finally:
        send_line('finish')
