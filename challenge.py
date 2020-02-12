import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn import svm


X0 = np.genfromtxt("./data/Xtr0_mat100.csv", delimiter=" ")
X1 = np.genfromtxt("./data/Xtr1_mat100.csv", delimiter=" ")
X2 = np.genfromtxt("./data/Xtr2_mat100.csv", delimiter=" ")

Y0 = np.genfromtxt("./data/Ytr0.csv", delimiter=",")[1:, 1]
Y1 = np.genfromtxt("./data/Ytr1.csv", delimiter=",")[1:, 1]
Y2 = np.genfromtxt("./data/Ytr2.csv", delimiter=",")[1:, 1]

X = np.concatenate((X0, X1), axis=0)
Y = np.concatenate((Y0, Y1))

val_X = X2
val_Y = Y2


test0 = np.genfromtxt("./data/Xte0_mat100.csv", delimiter=" ")
test1 = np.genfromtxt("./data/Xte1_mat100.csv", delimiter=" ")
test2 = np.genfromtxt("./data/Xte2_mat100.csv", delimiter=" ")
test = np.concatenate((test0, test1, test2), axis=0)

# kernel


def gaussianKernelGramMatrixFull(X1, X2, sigma=0.1):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(
                -np.sum(np.power((x1 - x2), 2)) / float(2 * (sigma ** 2))
            )
    return gram_matrix


# Library soultion
# clf = MLPClassifier(solver='lbfgs',learning_rate="adaptive", alpha=1e-5,hidden_layer_sizes=(1000,500,250,100, 80,80,80,80,80,80,80,80,80,80,80,80,40,20,10,5), random_state=1)
# clf.fit(X,Y)


# C=0.1
# kernel = gaussianKernelGramMatrixFull(X,X)
# kernel_test = gaussianKernelGramMatrixFull(test,test)
# clf = svm.SVC(C = C, kernel="precomputed")
# model = clf.fit(kernel, Y )
# model.predict(test)


def pesagos(x, y, rate, max_it):
    nb_sample = len(x)
    w = np.zeros(len(x[1]))
    step = 0
    for t in range(1, max_it):
        i = np.random.randint(0, nb_sample)
        eta = 1 / (rate * t)
        if (y[i] * w.dot(x[i])) < 1:
            w = (1 - eta * rate) * w + (rate * y[i]) * x[i]
        else:
            w = (1 - eta * rate) * w
    return w


def fit(w, x, y):
    nb_true = 0
    for i in range(0, len(x)):
        c = w.dot(x[i])
        print(c)
        if c < 0:
            c = 0
        else:
            c = 1
        if y[i] == c:
            nb_true += 1
    return nb_true / len(x)



# https://www.datasciencecentral.com/profiles/blogs/implementing-pegasos-primal-estimated-sub-gradient-solver-for-svm
def kernel_pesagos(K, y, rate, max_it):
    nb_sample = K.shape[0]
    alpha = np.zeros(X.shape[0])
    new_alpha = np.zeros(nb_sample)
    for t in range(1, max_it):
        i = np.random.randint(0, nb_sample)
        for j in range(0, nb_sample):
            if j != i:
                new_alpha[j] = alpha[j]
        sum = 0
        for j in range(0, nb_sample):
            sum += alpha[j] * y[j] * K[i, j]
        c = y[i] * (1 / rate * t) * sum
        if c < 1:
            new_alpha[i] = alpha[i] + 1
        else:
            new_alpha[i] = alpha[i]
        alpha = new_alpha
    return new_alpha



def kernel_fit(w, x, y = None):
    nb_true = 0
    nb_sample = x.shape[0]
    res = np.zeros(x.shape[0])
    for i in range(0, len(x)):
        sum = 0
        for j in range(0, nb_sample):
            sum += w[j] * x[i, j]
        print(sum)
        if sum < 1000:
            res[i]=1
        if y is not None:
           print(res[i], y[i])
           if res[i] == y[i]:
                nb_true += 1
    if y is not None:
        print("score ", nb_true / len(x))
    return res




def csv_pred(preds):
    with open("prediction.csv", "w") as f:
        f.write("Id,Bound\n")
        i = 0
        for pred in preds:
            f.write(str(i) + "," + str(int(pred)) + "\n")
            i += 1



## SIMPLE DATA ##

# pesagos

size = 500
X_simple = np.concatenate((np.random.rand(size,5), -np.random.rand(size,5) )) 
Y_simple = np.concatenate((np.ones(size), np.zeros(size) ))    


np.random.seed(10)
X_simple2 = np.concatenate((np.random.rand(size,5), -np.random.rand(size,5) )) 
Y_simple2 = np.concatenate((np.ones(size), np.zeros(size) ))    
w = pesagos(X_simple, Y_simple, 1000, 1000 )
fit(w, X_simple2 , Y_simple2)

#kernel pesagos

kernel_X_simple = gaussianKernelGramMatrixFull(X_simple, X_simple)
kernel_X_simple2 = gaussianKernelGramMatrixFull(X_simple2, X_simple2)


a = kernel_pesagos(kernel_X_simple, Y, 1000, 100000)
y = kernel_fit(a, kernel_X_simple2, np.random.randint(0,2,1000) )



## 


kernel_X = gaussianKernelGramMatrixFull(X1, X1)
kernel_val = gaussianKernelGramMatrixFull(X2,X2)

np.random.randint(0,2,1000) 
alpha = kernel_pesagos(kernel_X, Y1, 1000, 10000)
y = kernel_fit(alpha, kernel_val, Y2)