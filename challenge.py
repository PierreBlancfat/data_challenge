import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn import svm


X = np.genfromtxt('./data/Xtr0_mat100.csv', delimiter=' ')
Y = np.genfromtxt('./data/Ytr0.csv', delimiter=',')[1:, 1]

test = np.genfromtxt('./data/Xtr1_mat100.csv', delimiter=' ')
truth = np.genfromtxt('./data/Ytr1.csv', delimiter=',')[1:, 1]
clf = MLPClassifier(solver='lbfgs',learning_rate="adaptive", alpha=1e-5,hidden_layer_sizes=(1000,500,250,100, 80,80,80,80,80,80,80,80,80,80,80,80,40,20,10,5), random_state=1)
clf.fit(X,Y)

svn =  svm.SVC(gamma='auto')
svn.fit(X,Y)
svn.score(test,truth)

rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier(max_iter=5)
clf.fit(X_features, Y)
clf.score(test, truth)




def pesagos(x, y, rate, max_it):
    nb_sample = len(x)
    w = np.zeros(len(x[1]))
    step = 0
    for t in range(0, max_it):
        i = np.random.randint(0,nb_sample)
        if  y[i] * w.dot(x[i]) < 1:
            w = (1-rate)*w + rate*y[i]*x[i]
        else:
            w = (1-rate)*w
    return w

def fit(w, x, y):
    nb_true = 0
    for i in range(0,len(x)):
        c = w.dot(x[i])
        print(c)
        if c < 0 :
            c = 0
        else:
            c = 1
        if y[i] == c:
            nb_true += 1
    return nb_true/len(x)
 
print(fit(pesagos(X,Y,0.01, 10000), test, truth))

