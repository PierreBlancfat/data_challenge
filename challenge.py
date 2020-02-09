import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn import svm


X0 = np.genfromtxt('./data/Xtr0_mat100.csv', delimiter=' ')
X1 = np.genfromtxt('./data/Xtr1_mat100.csv', delimiter=' ')
X2 = np.genfromtxt('./data/Xtr2_mat100.csv', delimiter=' ')

Y0 = np.genfromtxt('./data/Ytr0.csv', delimiter=',')[1:, 1]
Y1 = np.genfromtxt('./data/Ytr1.csv', delimiter=',')[1:, 1]
Y2 = np.genfromtxt('./data/Ytr2.csv', delimiter=',')[1:, 1]

X =np.concatenate((X0, X1), axis=0)
test = X2
Y =np.concatenate((Y0, Y1, Y2))


test = np.genfromtxt('./data/Xte0_mat100.csv', delimiter=' ')
test1 = np.genfromtxt('./data/Xte1_mat100.csv', delimiter=' ')
test2 = np.genfromtxt('./data/Xte2_mat100.csv', delimiter=' ')
X =np.concatenate((test0, test1, test2), axis=1)

clf = MLPClassifier(solver='lbfgs',learning_rate="adaptive", alpha=1e-5,hidden_layer_sizes=(1000,500,250,100, 80,80,80,80,80,80,80,80,80,80,80,80,40,20,10,5), random_state=1)
clf.fit(X,Y)

test = np.concatenate((test,test1), axis =1)
truth = np.concatenate((truth,truth1))


def gaussianKernelGramMatrixFull(X1, X2, sigma=0.1):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum( np.power((x1 - x2),2) ) / float( 2*(sigma**2) ) )
    return gram_matrix

C=0.1
kernel = gaussianKernelGramMatrixFull(X,X)
kernel_test = gaussianKernelGramMatrixFull(test,test)
clf = svm.SVC(C = C, kernel="precomputed")
model = clf.fit(kernel, Y )
model.predict(test)


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
 


def csv_pred(preds):
    with open("prediction.csv","w") as f:
        f.write("Id,Bound\n")
        i = 0
        for pred in preds:
            f.write(str(i)+","+str(int(pred))+"\n")
            i += 1
    
print(fit(pesagos(X,Y,0.01, 10000), test, truth))

