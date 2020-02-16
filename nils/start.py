import numpy as np
import pandas as pd
from svm import SVM
from functools import partial


def create_submission(p):
    def predict_dataset(set_num, p):
        svm = SVM(
            partial(p_spectral_kernel_proj, p), 4**p, 1, data_sets[0][set_num],
            data_sets[1][set_num], data_sets[2][set_num])
        svm.fit(1)

        result = svm.predict_Z()
        for i in range(len(result)):
            if result[i] == -1:
                result[i] = 0

        return np.array([result]).T

    preds = np.vstack((predict_dataset(0, p), predict_dataset(1, p),
                       predict_dataset(2, p)))

    preds = np.hstack((np.array([np.arange(3000)]).T, preds))

    np.savetxt(
        'Yte.csv',
        preds,
        delimiter=',',
        fmt='%1i',
        header='Id,Bound',
        comments='')


data_path = '../data/'

Xtr0 = pd.read_csv(data_path + 'Xtr0.csv').values[:, 1]
Xte0 = pd.read_csv(data_path + 'Xte0.csv').values[:, 1]
Ytr0 = pd.read_csv(data_path + 'Ytr0.csv').values[:, 1]

for i in range(len(Ytr0)):
    if Ytr0[i] == 0:
        Ytr0[i] = -1

Xtr1 = pd.read_csv(data_path + 'Xtr1.csv').values[:, 1]
Xte1 = pd.read_csv(data_path + 'Xte1.csv').values[:, 1]
Ytr1 = pd.read_csv(data_path + 'Ytr1.csv').values[:, 1]

for i in range(len(Ytr1)):
    if Ytr1[i] == 0:
        Ytr1[i] = -1

Xtr2 = pd.read_csv(data_path + 'Xtr2.csv').values[:, 1]
Xte2 = pd.read_csv(data_path + 'Xte2.csv').values[:, 1]
Ytr2 = pd.read_csv(data_path + 'Ytr2.csv').values[:, 1]

for i in range(len(Ytr2)):
    if Ytr2[i] == 0:
        Ytr2[i] = -1

data_sets = [[Xtr0, Xtr1, Xtr2], [Ytr0, Ytr1, Ytr2], [Xte0, Xte1, Xte2]]


def p_spectral_kernel_proj(p, seq):
    def index(st):
        st_num = np.empty((p))
        for i in range(p):
            if st[i] == 'A':
                st_num[i] = 0
            if st[i] == 'C':
                st_num[i] = 1
            if st[i] == 'G':
                st_num[i] = 2
            if st[i] == 'T':
                st_num[i] = 3
        res = 0
        for i in range(p):
            res = res * 4 + st_num[i]
        return int(res)

    seq_vect = np.zeros((4**p))
    for i in range(len(seq) - p + 1):
        seq_vect[index(seq[i:i + p])] += 1
    return seq_vect


create_submission(7)

# p = 3
# ratio_train_test = 2 / float(3)
# set_num = 0
# mu = 1
# svm = SVM(partial(p_spectral_kernel_proj,p), 4**p, mu, data_sets[0][set_num], data_sets[1][set_num], data_sets[2][set_num])
# svm.fit(ratio_train_test)
# if ratio_train_test < 1:
#     ca = svm.categorization_accuracy()
#     print("Categorization accuracy: " + str(ca) + ' with p=' + str(p) + ' with data set ' + str(set_num))
# else:
#     predictions = svm.predict_Z()
#     print(predictions)
