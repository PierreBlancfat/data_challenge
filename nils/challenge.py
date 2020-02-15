import numpy as np
import pandas as pd
import os
from svm import SVM

data_path = '../data/'

Xtr0 = pd.read_csv(data_path + 'Xtr0.csv').values[:,1]
Xte0 = pd.read_csv(data_path + 'Xte0.csv').values[:,1]
Ytr0 = pd.read_csv(data_path + 'Ytr0.csv').values[:,1]

for i in range(len(Ytr0)):
    if Ytr0[i] == 0:
        Ytr0[i] = -1


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


l = len(Xtr0)

if os.path.isfile('one_spectrum_matrix.npy'):
    one_spectrum_matrix = np.load('one_spectrum_matrix.npy')
else:
    count = 0
    one_spectrum_matrix = np.empty((l, l))
    for i in range(l):
        for j in range(l):
            one_spectrum_matrix[i,j] = one_spectrum_kernel(Xtr0[i], Xtr0[j])
            count += 1
            if count % (l * l / 100) == 0:
                print(str(count / (l * l / 100)) + '% computed')
    np.save('one_spectrum_matrix.npy', one_spectrum_matrix)

svm = SVM(one_spectrum_matrix, one_spectrum_kernel, Xtr0, Ytr0, 1)
svm.fit()

print('Categorization accuracy: ' + str(svm.categorization_accuracy_on_train()))
print(svm.predict(Xtr0[0:5]))
