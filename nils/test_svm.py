import numpy as np
from svm import SVM

X = np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
Y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

matrix = np.empty((8,8))
for i in range(8):
    for j in range(8):
        matrix[i,j] = np.dot(X[i],X[j])

svm = SVM(matrix, np.dot, X, Y, 0)
svm.fit()

print('Categorization accuracy: ' + str(svm.categorization_accuracy_on_train()))

pred = np.array([[2, 0],[1,-2],[0,2],[0,-2]])

print(svm.predict(pred))
