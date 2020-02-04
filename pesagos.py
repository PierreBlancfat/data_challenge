import numpy as np


class Pesagos: 

    def __init__(self):
        self.w = None
    
    def train(self, x, y, rate, max_it):
        nb_sample = len(x)
        self.w = np.zeros(len(x[1]))
        step = 0
        for t in range(0, max_it):
            i = np.random.randint(0,nb_sample)
            if  y[i] * self.w.dot(x[i]) < 1:
                self.w = (1-rate)*self.w + rate*y[i]*x[i]
            else:
                self.w = (1-rate)*self.w
        return self.w

    def fit(self, x, y):
        nb_true = 0
        for i in range(0,len(x)):
            c = self.w.dot(x[i])
            if c < 0 :
                c = 0
            else:
                c = 1
            if y[i] == c:
                nb_true += 1
        print(nb_true/len(x))
