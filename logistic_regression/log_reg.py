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
        return ret

    def learn(self, X, y, l=1):
        self.theta = np.zeros(np.size(self.X[0,:]))

        def sigmoid(Z):
            return (1 / (1 + np.exp(-1 * Z)))

        def costFunctionReg(X, y, theta, l):
            m = np.size(X[:,0])
            h = sigmoid(np.matmul(X, theta))
            h = np.matrix(h)
            j = 1 / m * np.sum(np.matmul(-1 * y.T, np.log(h)) - np.matmul((1 - y).T, np.log(1 - h)))
            j = j + (l / (2 * m) * np.sum(np.power(theta[1,np.size(theta[:,1:])], 2)))
            j = np.matrix(j)

            grad =  1 / m * np.matmul((h - y).T, X)
            grad[1:] = grad[1:] + l / m * theta[1,np.size(theta[:,1:])].T

            return j.flatten(), grad

        #print(costFunctionReg(self.X, self.y, self.theta, l))

        def ft_reg(theta):
            return costFunctionReg(self.X, self.y, theta, l)

        print(fmin_bfgs(ft_reg, x0=self.theta, maxiter=400))

if __name__ == "__main__":
    data = pd.read_csv("ex2data2.txt")
    data = np.matrix(data, float)
    X = data[:,[0,1]]
    y = data[:,[2]]

    model = log_reg(X, y)
