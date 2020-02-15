import numpy as np
import cvxopt as co


class SVM:
    def __init__(self, kernel_proj, kp_size, mu, X, Y, Z):
        self.kernel_proj = kernel_proj
        self.X = X
        self.Y = Y
        self.mu = mu
        self.Z = Z
        self.kp_size = kp_size
        self.feat_X = self.compute_features(X)
        self.feat_Z = None
        self.ps_X_X = self.compute_kernel(self.feat_X, self.feat_X)
        self.ps_X_Z = None

    def compute_features(self, X):
        print('Computing ' + str(len(X)) + ' features of size ' +
              str(self.kp_size))
        features = np.empty((len(X), self.kp_size))
        for i in range(len(X)):
            features[i, :] = self.kernel_proj(X[i])
        print('Computation over')
        return features

    def compute_kernel(self, feat_X, feat_Y):
        size_X = len(feat_X[:, 0])
        size_Y = len(feat_Y[:, 0])
        print('Computing kernel of size ' + str(size_X) + ' * ' + str(size_Y))
        kernel = np.empty((size_X, size_Y))
        for i in range(min(size_X, size_Y)):
            for j in range(i):
                ps = np.dot(feat_X[i], feat_Y[j])
                kernel[i, j] = ps
                kernel[j, i] = ps
        for i in range(min(size_X, size_Y)):
            kernel[i, i] = np.dot(feat_X[i], feat_Y[i])
        if size_X > size_Y:
            for i in range(size_Y, size_X):
                for j in range(size_Y):
                    kernel[i, j] = np.dot(feat_X[i], feat_Y[j])
        else:
            for i in range(size_X):
                for j in range(size_X, size_Y):
                    kernel[i, j] = np.dot(feat_X[i], feat_Y[j])
        print('Computation over')
        return kernel

    def fit(self, ratio_train_test):
        self.l_train = int(len(self.Y) * ratio_train_test)

        P = np.empty((self.l_train, self.l_train))
        for i in range(self.l_train):
            for j in range(self.l_train):
                P[i, j] = self.Y[i] * self.Y[j]
                if i == j:
                    P[i, j] *= (self.mu + self.ps_X_X[i, j])
                else:
                    P[i, j] *= self.ps_X_X[i, j]
        P = 2 * co.matrix(P)

        q = co.matrix(np.zeros((self.l_train)))

        G = co.matrix(-1 * np.identity(self.l_train))

        h = co.matrix(np.zeros((self.l_train)))

        A = np.zeros((2, self.l_train))
        b = np.zeros((2))
        for i in range(self.l_train):
            A[0, i] = 1
        b[0] = 1
        for i in range(self.l_train):
            A[1, i] = self.Y[i]
        b[1] = 0
        A = co.matrix(A)
        b = co.matrix(b)

        self.A_star = co.solvers.qp(P, q, G, h, A, b)

        self.solution = np.array(self.A_star['x'])

        self.gamma_star = 0
        for i in range(self.l_train):
            for j in range(self.l_train):
                temp = self.solution[i] * self.solution[j] * self.Y[
                    i] * self.Y[j]
                if i == j:
                    temp *= (self.mu + self.ps_X_X[i, j])
                else:
                    temp *= self.ps_X_X[i, j]
                self.gamma_star += temp
        self.gamma_star = np.sqrt(self.gamma_star)

        i = 0
        while self.solution[i] <= 0:
            i += 1

        self.b = self.Y[i] * self.gamma_star**2
        temp = 0
        for j in range(self.l_train):
            temp2 = self.ps_X_X[i, j]
            if i == j:
                temp2 += self.mu
            temp += self.solution[j] * self.Y[j] * temp2
        self.b -= temp

    def categorization_accuracy(self):
        correct = 0
        preds = self.predict_test()
        for i in range(len(self.Y) - self.l_train):
            if preds[i] == self.Y[i + self.l_train]:
                correct += 1
        return correct / float(len(preds))

    def predict_test(self):
        Y_test = np.zeros((len(self.Y) - self.l_train))
        for i in range(len(self.Y) - self.l_train):
            for j in range(self.l_train):
                Y_test[i] += self.solution[j] * self.Y[j] * self.ps_X_X[
                    j, i + self.l_train]
            Y_test[i] += self.b
            Y_test[i] = np.sign(Y_test[i])
            print(str(i + 1) + ' predictions over ' + str(len(Y_test)))
        return Y_test.astype(int)

    def predict_Z(self):
        self.feat_Z = self.compute_features(self.Z)
        self.ps_X_Z = self.compute_kernel(self.feat_X, self.feat_Z)
        Y = np.zeros((len(self.Z)))
        for i in range(len(self.Z)):
            for j in range(self.l_train):
                Y[i] += self.solution[j] * self.Y[j] * self.ps_X_Z[j, i]
            Y[i] += self.b
            Y[i] = np.sign(Y[i])
            print(str(i + 1) + ' predictions over ' + str(len(Y)))
        return Y.astype(int)
