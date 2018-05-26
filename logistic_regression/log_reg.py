import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs

class log_reg(object):
    """docstring for log_reg."""
    def __init__(self, X, y):
        super(log_reg, self).__init__()
        self.X = np.matrix(X)
        self.y = y.reshape(-1,1)
        self.X = self.mapFeature(self.X[:,0], self.X[:,1])
        self.learn(self.X, self.y)

    def featureNormalize(self, X):
        mu = X.mean(axis = 0)
        sigma = X.std(axis = 0)
        self.X = (X - mu) / sigma
        return self.X, mu, sigma

    def mapFeature(self, X1, X2, deg=6):
        ret = np.ones((np.size(X1[:,0]), 1))
        for i in range(1, deg + 1):
            for j in range(0, i + 1):
                ret = np.c_[ret, np.multiply(np.power(X1, i - j), np.power(X2, j))]
        return np.matrix(ret)

    def learn(self, X, y, l=1):
        self.theta = np.matrix(np.zeros(X.shape[1]))

        def sigmoid(Z):
            return np.matrix((1.0 / (1.0 + np.exp(-1.0 * Z))))

        def costFunction(theta, X, y, l):
            m = np.size(X[:,0])
            h = sigmoid(np.matmul(X, theta.T))
            a = np.asarray(-y) * np.asarray(np.log(sigmoid(np.matmul(X, theta.T))))
            b = np.asarray(1.0 - y) * np.asarray(np.log(1 - sigmoid(np.matmul(X, theta.T))))
            c = l / (2 * m) * np.sum(np.power(theta[0,1:], 2))
            J = 1.0 / m * np.sum(a - b) + c
            return J

        def gradFunction(theta, X, y, l):
            m = np.size(X[:,0])
            h = sigmoid(np.matmul(X, theta.T))
            a = h - y
            g = 1 / m * np.matmul(a.T, X)
            print(g)
            g[0,1:np.size(g)] = g[0,1:np.size(g)] + l / m * theta[0,1:]
            return g

        #print(costFunctionReg(self.X, self.y, self.theta, l))

        def f(theta):
            return np.ndarray.flatten(costFunction(theta, X, y, 1.0))

        def fprime(theta):
            return np.ndarray.flatten(gradFunction(theta, X, y, 1.0))

        print(fmin_bfgs(f, self.theta, fprime, maxiter=400))
        #print(fmin_bfgs(ft_reg, self.theta, maxiter=400))
        #print(ft_reg(self.initial_theta))

if __name__ == "__main__":
    data = np.genfromtxt("ex2data2.txt", delimiter=',')
    data = np.matrix(data, float)
    X = data[:,[0,1]]
    y = data[:,[2]]

    model = log_reg(X, y)
