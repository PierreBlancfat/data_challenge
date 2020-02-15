import numpy as np
import pandas as pd
import os
from svm import SVM
from functools import partial

def create_submission():
    two_spectrum_matrix = compute_two_spec_mat(Xtr0)

    svm = SVM(two_spectrum_matrix, partial(k_spectrum_kernel,k=2), Xtr0, Ytr0, 1)
    svm.fit()

    print('Categorization accuracy: ' + str(svm.categorization_accuracy_on_train()))

    preds0 = np.array([svm.predict(Xte0, compute_two_spec_mat(Xtr0, Xte0))]).T

    two_spectrum_matrix = compute_two_spec_mat(Xtr1)

    svm = SVM(two_spectrum_matrix, partial(k_spectrum_kernel,k=2), Xtr1, Ytr1, 1)
    svm.fit()

    print('Categorization accuracy: ' + str(svm.categorization_accuracy_on_train()))

    preds1 = np.array([svm.predict(Xte1, compute_two_spec_mat(Xtr1, Xte1))]).T

    two_spectrum_matrix = compute_two_spec_mat(Xtr2)

    svm = SVM(two_spectrum_matrix, partial(k_spectrum_kernel,k=2), Xtr2, Ytr2, 1)
    svm.fit()

    print('Categorization accuracy: ' + str(svm.categorization_accuracy_on_train()))

    preds2 = np.array([svm.predict(Xte2, compute_two_spec_mat(Xtr2, Xte2))]).T

    preds = np.vstack((preds0, preds1, preds2))

    preds = np.hstack((np.array([np.arange(3000)]).T, preds))

    np.savetxt('submission.csv', preds, delimiter=',', fmt='%1i', header='Id,Bound', comments='')

data_path = '../data/'

Xtr0 = pd.read_csv(data_path + 'Xtr0.csv').values[:,1]
Xte0 = pd.read_csv(data_path + 'Xte0.csv').values[:,1]
Ytr0 = pd.read_csv(data_path + 'Ytr0.csv').values[:,1]

for i in range(len(Ytr0)):
    if Ytr0[i] == 0:
        Ytr0[i] = -1

Xtr1 = pd.read_csv(data_path + 'Xtr1.csv').values[:,1]
Xte1 = pd.read_csv(data_path + 'Xte1.csv').values[:,1]
Ytr1 = pd.read_csv(data_path + 'Ytr1.csv').values[:,1]

for i in range(len(Ytr1)):
    if Ytr1[i] == 0:
        Ytr1[i] = -1

Xtr2 = pd.read_csv(data_path + 'Xtr2.csv').values[:,1]
Xte2 = pd.read_csv(data_path + 'Xte2.csv').values[:,1]
Ytr2 = pd.read_csv(data_path + 'Ytr2.csv').values[:,1]

for i in range(len(Ytr2)):
    if Ytr2[i] == 0:
        Ytr2[i] = -1


def count_nucl(x):
    c = np.zeros((4)) # Dans l'ordre: A, C, G, T
    
    for i in range(len(x)):
        if x[i] == 'A':
            c[0] += 1
        elif x[i] == 'C':
            c[1] += 1
        elif x[i] == 'G':
            c[2] += 1
        elif x[i] == 'T':
            c[3] += 1
        else:
            print('ERROR: unknown nucleotid')
            
    return c

def one_spectrum_kernel(x, z):
    c_x = count_nucl(x)
    c_z = count_nucl(z)
    
    return np.dot(c_x, c_z)

def k_suffix_kernel(x, z, k):
    for i in range(k):
        if x[-i - 1] != z[-i-1]:
            return 0
    return 1

def k_spectrum_kernel(x,z,k):
    total = 0
    for i in range(len(x)-k):
        for j in range(len(z)-k):
            total += k_suffix_kernel(x[i:i+k],z[j:j+k],k)
    return total

def compute_two_spec_mat(X,Pred=[]):
    def index(st):
        if st[0] == 'A':
            if st[1] == 'A':
                return 0
            if st[1] == 'C':
                return 1
            if st[1] == 'G':
                return 2
            if st[1] == 'T':
                return 3
        if st[0] == 'C':
            if st[1] == 'A':
                return 4
            if st[1] == 'C':
                return 5
            if st[1] == 'G':
                return 6
            if st[1] == 'T':
                return 7
        if st[0] == 'G':
            if st[1] == 'A':
                return 8
            if st[1] == 'C':
                return 9
            if st[1] == 'G':
                return 10
            if st[1] == 'T':
                return 11
        if st[0] == 'T':
            if st[1] == 'A':
                return 12
            if st[1] == 'C':
                return 13
            if st[1] == 'G':
                return 14
            if st[1] == 'T':
                return 15
    if len(Pred) == 0:
        X_arr = np.zeros((len(X), 16))
        for j in range(len(X)):
            for i in range(len(X[0]) - 1):
                X_arr[j,index(X[j][i:i+2])] += 1
        mat = np.empty((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                mat[i,j] = np.dot(X_arr[i], X_arr[j])
        return mat
    else:
        X_arr = np.zeros((len(X), 16))
        Pred_arr = np.zeros((len(Pred), 16))
        for j in range(len(X)):
            for i in range(len(X[0]) - 1):
                X_arr[j,index(X[j][i:i+2])] += 1
        for j in range(len(Pred)):
            for i in range(len(Pred[0]) - 1):
                Pred_arr[j,index(Pred[j][i:i+2])] += 1                
        mat = np.empty((len(X), len(Pred)))
        for i in range(len(X)):
            for j in range(len(Pred)):
                mat[i,j] = np.dot(X_arr[i], Pred_arr[j])
        return mat

def compute_n_spec_mat(X,n,Pred=[]):
    def index(st):
        st_num = np.empty((n))
        for i in range(n):
            if st[i] == 'A':
                st_num[i] = 0
            if st[i] == 'C':
                st_num[i] = 1
            if st[i] == 'G':
                st_num[i] = 2
            if st[i] == 'T':
                st_num[i] = 3
        res = 0
        for i in range(n):
            res = res * 4 + st_num[i]
        return int(res)

    if len(Pred) == 0:
        X_arr = np.zeros((len(X), 4**n))
        for j in range(len(X)):
            for i in range(len(X[0]) - n + 1):
                X_arr[j,index(X[j][i:i+n])] += 1
        mat = np.empty((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                mat[i,j] = np.dot(X_arr[i], X_arr[j])
        return mat
    else:
        X_arr = np.zeros((len(X), 4**n))
        Pred_arr = np.zeros((len(Pred), 4**n))
        for j in range(len(X)):
            for i in range(len(X[0]) - n + 1):
                X_arr[j,index(X[j][i:i+n])] += 1
        for j in range(len(Pred)):
            for i in range(len(Pred[0]) -n + 1):
                Pred_arr[j,index(Pred[j][i:i+n])] += 1                
        mat = np.empty((len(X), len(Pred)))
        for i in range(len(X)):
            for j in range(len(Pred)):
                mat[i,j] = np.dot(X_arr[i], Pred_arr[j])
        return mat        
    

l = len(Xtr0)

# if os.path.isfile('one_spectrum_matrix.npy'):
#     one_spectrum_matrix = np.load('one_spectrum_matrix.npy')
# else:
#     count = 0
#     one_spectrum_matrix = np.empty((l, l))
#     for i in range(l):
#         for j in range(l):
#             one_spectrum_matrix[i,j] = one_spectrum_kernel(Xtr0[i], Xtr0[j])
#             count += 1
#             if count % (l * l / 100) == 0:
#                 print(str(count / (l * l / 100)) + '% computed')
#     np.save('one_spectrum_matrix.npy', one_spectrum_matrix)

#create_submission()

p = 6

p_spectrum_matrix = compute_n_spec_mat(Xtr0, n=p)

svm = SVM(p_spectrum_matrix, partial(k_spectrum_kernel,k=p), Xtr0, Ytr0, 1)
svm.fit()

print('With p=' + str(p) + ', Categorization accuracy: ' + str(svm.categorization_accuracy_on_train()))

#preds = svm.predict(Xte0, compute_n_spec_mat(Xtr0, n=p, Pred=Xte0))

