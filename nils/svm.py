import numpy as np
import cvxopt as co

class SVM:
    def __init__(self, kernel, kernel_function, X, Y, mu):
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.l = len(Y)
        self.mu = mu
        self.kernel_function = kernel_function

    def fit(self):
        P = np.empty((self.l, self.l))
        for i in range(self.l):
            for j in range(self.l):
                P[i,j] = self.Y[i] * self.Y[j]
                if i == j:
                    P[i,j] *= (self.mu + self.kernel[i,j])
                else:
                    P[i,j] *= self.kernel[i,j]
        P = 2 * co.matrix(P)

        q = co.matrix(np.zeros((self.l)))

        G = co.matrix(-1 * np.identity(self.l))

        h = co.matrix(np.zeros((self.l)))

        A = np.zeros((2, self.l))
        b = np.zeros((2))
        for i in range(self.l):
            A[0,i] = 1
        b[0] = 1
        for i in range(self.l):
            A[1,i] = self.Y[i]
        b[1] = 0
        A = co.matrix(A)
        b = co.matrix(b)

        self.A_star = co.solvers.qp(P, q, G, h, A, b)

        self.solution = np.array(self.A_star['x'])

        self.gamma_star = 0
        for i in range(self.l):
            for j in range(self.l):
                temp = self.solution[i] * self.solution[j] * self.Y[i] * self.Y[j]
                if i == j:
                    temp *= (self.mu + self.kernel[i,j])
                else:
                    temp *= self.kernel[i,j]
                self.gamma_star += temp
        self.gamma_star = np.sqrt(self.gamma_star)

        i = 0
        while self.solution[i] <= 0:
            i += 1

        self.b = self.Y[i] * self.gamma_star**2
        temp = 0
        for j in range(self.l):
            temp2 = self.kernel[i,j]
            if i == j:
                temp2 += self.mu
            temp += self.solution[j] * self.Y[j] * temp2
        self.b -= temp

    def categorization_accuracy_on_train(self):
        pred = np.zeros((self.l))
        for i in range(self.l):
            for j in range(self.l):
                pred[i] += self.solution[j] * self.Y[j] * self.kernel[j,i]
            pred[i] += self.b
            pred[i] = np.sign(pred[i])

        correct = 0
        for i in range(self.l):
            if pred[i] == self.Y[i]:
                correct += 1
        return correct / float(self.l)
    
    def predict(self, X, matrix=[]):
        lenX = len(X)
        Y = np.zeros((lenX))
        for i in range(lenX):
            for j in range(self.l):
                if len(matrix) == 0:
                    Y[i] += self.solution[j] * self.Y[j] * self.kernel_function(self.X[j], X[i])
                else:
                    Y[i] += self.solution[j] * self.Y[j] * matrix[j,i]
            Y[i] += self.b
            Y[i] = np.sign(Y[i])
            print(str(i+1) + ' predictions over ' + str(lenX))
        for i in range(lenX):
            if Y[i] == -1:
                Y[i] = 0
        return Y.astype(int)
