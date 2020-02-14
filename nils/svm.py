import numpy as np
import cvxopt as co

class SVM:
    def __init__(self, kernel, X, Y, mu):
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.l = len(Y)
        self.mu = mu

    def fit(self):
        P = np.empty((self.l, self.l))
        for i in range(self.l):
            for j in range(self.l):
                P[i,j] = self.Y[i] * self.Y[j] * self.kernel[i,j]
                if i == j:
                    P[i,j] += self.mu
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
