import os

import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import re
from scipy import stats
from natsort import natsorted
from PIL import Image
import statistics

import parameter


class plot:
    def __init__(self):
        self.up = [0, 0, 0, 0, 0]
        self.down = [0, 0, 0, 0, 0]
        self.b = 0
        self.l = 0

    def plot_operate(self, smell_name, project_name):
        target = 'MCC'
        # [Bayes_baseline, Bayes_noise] -> scatter and box, smell
        Bayes_baseline = self.import_data(smell_name, project_name, alg='Bayes_baseline')
        Bayes_noise = self.import_data(smell_name, project_name, alg='Bayes_noisy')
        Bayes_pseudo = self.import_data(smell_name, project_name, alg='Bayes_pseudo')
        self.box_plot((Bayes_baseline, Bayes_noise, Bayes_pseudo), ['baseline', 'A1', 'A2'],
                      project_name, smell_name, target, 'Bayes_compare')

        Logistic_baseline = self.import_data(smell_name, project_name, alg='Logistic_baseline')
        Logistic_noise = self.import_data(smell_name, project_name, alg='Logistic_noisy')
        Logistic_pseudo = self.import_data(smell_name, project_name, alg='Logistic_pseudo')
        self.box_plot((Logistic_baseline, Logistic_noise, Logistic_pseudo), ['baseline', 'A1', 'A2'],
                      project_name, smell_name, target, 'Logistic_compare')

        Bayes_add_baseline = self.import_data(smell_name, project_name, alg='add_out_bayes_baseline')
        Bayes_add_noise = self.import_data(smell_name, project_name, alg='add_out_bayes_noisy')
        Bayes_add_pseudo = self.import_data(smell_name, project_name, alg='add_out_bayes_pseudo')
        self.box_plot((Bayes_baseline, Bayes_add_baseline),
                      ['baseline', 'add_baseline'],
                      project_name, smell_name, target, 'Bayes_add_out')
        self.box_plot((Bayes_baseline, Bayes_add_baseline, Bayes_add_noise, Bayes_add_pseudo),
                      ['baseline', 'add_baseline', 'add_A1', 'add_A2'],
                      project_name, smell_name, target, 'Bayes_add_clean')
        # box_plot((Bayes_baseline, Bayes_pseudo, Bayes_add_baseline, Bayes_add_pseudo),
        #          ['baseline', 'A2', 'add_baseline', 'add_A2'],
        #          project_name, smell_name, target, 'Bayes_add_pseudo')
        # box_plot((Bayes_noise, Bayes_pseudo, Bayes_add_noise, Bayes_add_pseudo),
        #          ['A1', 'A2', 'add_A1', 'add_A2'],
        #          project_name, smell_name, target, 'Bayes_add_noise_pseudo')

        Logistic_add_baseline = self.import_data(smell_name, project_name, alg='add_out_logistic_baseline')
        Logistic_add_noise = self.import_data(smell_name, project_name, alg='add_out_logistic_noisy')
        Logistic_add_pseudo = self.import_data(smell_name, project_name, alg='add_out_logistic_pseudo')
        self.box_plot((Logistic_baseline, Logistic_add_baseline),
                      ['baseline', 'add_baseline'],
                      project_name, smell_name, target, 'Logistic_add_out')
        self.box_plot((Logistic_baseline, Logistic_add_baseline, Logistic_add_noise, Logistic_add_pseudo),
                      ['baseline', 'add_baseline', 'add_A1', 'add_A2'],
                      project_name, smell_name, target, 'Logistic_add_clean')
        # box_plot((Logistic_baseline, Logistic_pseudo, Logistic_add_baseline, Logistic_add_pseudo),
        #          ['baseline', 'A2', 'add_baseline', 'add_A2'],
        #          project_name, smell_name, target, 'Logistic_add_pseudo')
        # box_plot((Logistic_noise, Logistic_pseudo, Logistic_add_noise, Logistic_add_pseudo),
        #          ['A1', 'A2', 'add_A1', 'add_A2'],
        #          project_name, smell_name, target, 'Logistic_add_noise_pseudo')

    def test_operate(self, smell_name, project_name, alg):
        target = 'MCC'
        baseline = self.import_data(smell_name, project_name, alg='{0}_baseline'.format(alg))
        noise = self.import_data(smell_name, project_name, alg='{0}_noisy'.format(alg))
        pseudo = self.import_data(smell_name, project_name, alg='{0}_pseudo'.format(alg))

        add_baseline = self.import_data(smell_name, project_name, alg='add_out_{0}_baseline'.format(alg.lower()))
        add_noise = self.import_data(smell_name, project_name, alg='add_out_{0}_noisy'.format(alg.lower()))
        add_pseudo = self.import_data(smell_name, project_name, alg='add_out_{0}_pseudo'.format(alg.lower()))

        print("{0} & {1} & {2} & {3} & {4} & {5}hhhlineh".format(project_name,
                                                                 self.one_t_test(baseline, add_baseline, 0),
                                                                 self.mannu(add_baseline, add_noise, 1),
                                                                 self.mannu(add_baseline, add_pseudo, 2),
                                                                 self.one_t_test(baseline, noise, 3),
                                                                 self.one_t_test(baseline, pseudo, 4)))

    def compare(self, smell_name, project_name):
        Bayes_baseline = self.import_data(smell_name, project_name, alg='Bayes_baseline')
        Logistic_baseline = self.import_data(smell_name, project_name, alg='Logistic_baseline')
        if Bayes_baseline.size == 0 or Logistic_baseline.size == 0:
            pass
        elif float(Bayes_baseline[0]) > float(Logistic_baseline[0]):
            self.b = self.b + 1
        elif float(Bayes_baseline[0]) < float(Logistic_baseline[0]):
            self.l = self.l + 1


    def compare2(self, smell_name, project_name):
        Bayes_baseline = self.import_data(smell_name, project_name, alg='Bayes_baseline')
        Logistic_noise = self.import_data(smell_name, project_name, alg='Logistic_noisy')
        Logistic_pseudo = self.import_data(smell_name, project_name, alg='Logistic_pseudo')

        if Logistic_noise.mean() > Logistic_pseudo.mean():
            print("{0} & {1}".format(project_name, self.one_t_test(Bayes_baseline, Logistic_noise, 0)))
        else:
            print("{0} & {1}".format(project_name, self.one_t_test(Bayes_baseline, Logistic_pseudo, 0)))




    def import_data(self, smell_name, project_name='ant', alg='Bayes', target='MCC'):
        result = np.array([])
        path_name = '{0}{1}/{2}/{3}.csv'.format(parameter.pred_path, smell_name, project_name, alg)
        match_path = glob.glob(path_name)
        array = pd.read_csv(match_path[0])[target]

        array = array.values
        result = np.concatenate([result, array])
        if result.size == 0:
            result = np.zeros(1)
        result = result[~(result == 0.0)]
        return result

    def box_plot(self, points, labels, project_name, smell_name, target, file_name):
        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.set_ylim(-1.0, 1.0)
        bp = ax.boxplot(points)
        plt.title('{0}'.format(smell_name))
        ax.set_xticklabels(labels)
        plt.ylabel('{0}'.format(project_name))
        plt.grid()

        if 'add_out' in file_name:
            create_dir('result', 'exp1')
            create_dir('result/exp1', smell_name)
            if 'Bayes' in file_name:
                create_dir('result/exp1/' + smell_name, 'Bayes')
                fig.savefig('./result/exp1/{0}/Bayes/{1}_{2}.png'.format(smell_name, project_name, file_name))
            elif 'Logistic' in file_name:
                create_dir('result/exp1/' + smell_name, 'Logistic')
                fig.savefig('./result/exp1/{0}/Logistic/{1}_{2}.png'.format(smell_name, project_name, file_name))

        elif 'add_clean' in file_name:
            create_dir('result', 'exp2')
            create_dir('result/exp2', smell_name)
            if 'Bayes' in file_name:
                create_dir('result/exp2/' + smell_name, 'Bayes')
                fig.savefig('./result/exp2/{0}/Bayes/{1}_{2}.png'.format(smell_name, project_name, file_name))
            elif 'Logistic' in file_name:
                create_dir('result/exp2/' + smell_name, 'Logistic')
                fig.savefig('./result/exp2/{0}/Logistic/{1}_{2}.png'.format(smell_name, project_name, file_name))

        elif 'compare' in file_name:
            create_dir('result', 'exp3')
            create_dir('result/exp3', smell_name)
            if 'Bayes' in file_name:
                create_dir('result/exp3/' + smell_name, 'Bayes')
                fig.savefig('./result/exp3/{0}/Bayes/{1}_{2}.png'.format(smell_name, project_name, file_name))
            elif 'Logistic' in file_name:
                create_dir('result/exp3/' + smell_name, 'Logistic')
                fig.savefig('./result/exp3/{0}/Logistic/{1}_{2}.png'.format(smell_name, project_name, file_name))

        plt.close()
        plt.close()

    def one_t_test(self, pop_mean, group, num):
        res = '- & -'
        if np.unique(group).size < 3:
            res = '※ & -'
        try:
            t, p = stats.ttest_1samp(group, pop_mean)
            if p.size == 0:
                return res
            if p[0] < 0.05:
                res = 'p< 0.05'
                if group.mean() > pop_mean[0]:
                    self.up[num] = self.up[num] + 1
                    res = res + ' & up'
                else:
                    self.down[num] = self.down[num] + 1
                    res = res + ' & down'
            elif p[0] >= 0.05:
                res = '0.05< p & -'
        finally:
            return res

    def mannu(self, a, b, num):
        res = '- & -'
        if np.unique(a).size < 3 and np.unique(b).size < 3:
            res = '※ & -'
            return res
        elif np.unique(a).size == 1:
            res = self.one_t_test(a, b, num)
            return res
        elif np.unique(b).size == 1:
            res = self.one_t_test(b, a, num)
            return res + '2'
        elif np.unique(a).size == 0 or np.unique(b).size == 0:
            return '※ & -'
        try:
            p = stats.mannwhitneyu(a, b).pvalue
            if p < 0.05:
                res = 'p< 0.05'
                if a.mean() < b.mean():
                    self.up[num] = self.up[num] + 1
                    res = res + ' & up'
                else:
                    self.down[num] = self.down[num] + 1
                    res = res + ' & down'
            elif p >= 0.05:
                res = '0.05< p & -'

        finally:
            return res


def montage(exp, smell_name, alg):
    x = 3
    y = 4
    files = glob.glob("./result/{0}/{1}/{2}/*.png".format(exp, smell_name, alg))
    d = []

    for i in natsorted(files):
        img = Image.open(i)
        img = np.asarray(img)
        d.append(img)

    fig, ax = plt.subplots(y, x, figsize=(10, 10))
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(y):
        for j in range(x):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(d[x * i + j], cmap="bone")
    plt.savefig("./result/{0}/{1}/{2}/concat.png".format(exp, smell_name, alg))
    plt.close()


def create_dir(a, b):
    if not os.path.isdir('./{0}/'.format(str(a))):
        os.mkdir('./{0}'.format(str(a)))
    if not os.path.isdir('./{0}/{1}'.format(str(a), str(b))):
        os.mkdir('./{0}/{1}/'.format(str(a), str(b)))


if __name__ == '__main__':
    # for smell_name in parameter.smellNames:
    #     for project_name in parameter.projectNames:
    #         p = re.findall('[a-z]+', project_name[0])[0]
    #         pl = plot()
    #         pl.plot_operate(smell_name, p)

    # for smell_name in parameter.smellNames:
    #     for alg in ['Bayes', 'Logistic']:
    #         print(smell_name)
    #         print(alg)
    #         pl = plot()
    #         for project_name in parameter.projectNames:
    #             p = re.findall('[a-z]+', project_name[0])[0]
    #             pl.test_operate(smell_name, p, alg)
    #         print('Total & & {0}:{1} & & {2}:{3} & & {4}:{5} & & {6}:{7} & & {8}:{9} hhhlineh'.format(pl.up[0], pl.down[0],
    #                                                                                         pl.up[1], pl.down[1],
    #                                                                                         pl.up[2], pl.down[2],
    #                                                                                         pl.up[3], pl.down[3],
    #                                                                                         pl.up[4], pl.down[4]))
    #         print('\n\n')

    # for smell_name in parameter.smellNames:
    #     print('\n\nsmell name: {0}'.format(smell_name))
    #     pl = plot()
    #     for project_name in parameter.projectNames:
    #         p = re.findall('[a-z]+', project_name[0])[0]
    #         pl.compare2(smell_name, p)
    #     print('Total && {0}:{1}'.format(pl.up[0], pl.down[0]))


    # for exp in ['exp1', 'exp2', 'exp3']:
    #     for smell_name in parameter.smellNames:
    #         for alg in ['Bayes', 'Logistic']:
    #             montage(exp, smell_name, alg)

    # for smell_name in parameter.smellNames:
    #     pl = plot()
    #     for project_name in parameter.projectNames:
    #         p = re.findall('[a-z]+', project_name[0])[0]
    #         pl.compare(smell_name, p)
    #     print('smell name: {0}'.format(smell_name))
    #     print('Bayes: {0}, Log: {1}'.format(pl.b, pl.l))
    #
    # for project_name in parameter.projectNames:
    #     pl = plot()
    #     for smell_name in parameter.smellNames:
    #         p = re.findall('[a-z]+', project_name[0])[0]
    #         pl.compare(smell_name, p)
    #     print('project name: {0}'.format(project_name[-1]))
    #     print('Bayes: {0}, Log: {1}'.format(pl.b, pl.l))

    # print('Smell & Min & Mean & Median & Max & Total hhhlineh')
    # for sm in parameter.smellNames:
    #     l = []
    #
    #     for p in parameter.projectNames:
    #         for i in p:
    #             index_name = pd.read_csv(parameter.return_path(sm, i)).columns.values[0].split('.')
    #
    #             df = pd.read_csv(parameter.return_path(sm, i), sep=',', skiprows=1,
    #                                     header=None, names=index_name, quoting=3).drop(columns='name').replace('(.*)"(.*)',
    #                                                                                                            r'\1\2',
    #                                                                                                            regex=True)
    #             y = df[['isSmelly']].replace('true', 1).replace('false', 0).values.reshape(-1)
    #             l.append(np.sum(y))
    #     print('{0} & {1} & {2} & {3} & {4} & {5} hhhlineh'.format(sm, min(l), statistics.mean(l), statistics.median(l), max(l), sum(l)))


######
    import main
    p = parameter.ant
    X_train, y_train, X_test, y_test = main.input_dataset(parameter.return_path('GodClass', p[0]),
                                                     parameter.return_paths('GodClass', p[1:-1]),
                                                     parameter.return_path('GodClass', p[-1]))
    import random
    from multiprocessing import cpu_count

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
    from cleanlab.classification import LearningWithNoisyLabels
    index = pd.read_csv(parameter.return_path('GodCLass', p[-1])).columns.values[0].split('.')
    df = pd.read_csv(parameter.return_path('GodClass', p[-1]), sep=',', skiprows=1,
                      header=None, names=index, quoting=3).drop(columns='name').replace('(.*)"(.*)', r'\1\2',
                                                                                             regex=True)
    model = GaussianNB()
    clf = LearningWithNoisyLabels(clf=model, seed=1, n_jobs=cpu_count())
    clf.fit(X_train, y_train)
    # X_with_noise = df[:][clf.noise_mask]
    print(type(clf.noise_mask))
    print(np.where(clf.noise_mask == True))
    print('\n\n')
    model = LogisticRegression(multi_class='auto')
    clf = LearningWithNoisyLabels(clf=model, seed=1, n_jobs=cpu_count())
    clf.fit(X_train, y_train)
    # X_with_noise2 = df[:][clf.noise_mask]
    print(np.where(clf.noise_mask == True))
#####