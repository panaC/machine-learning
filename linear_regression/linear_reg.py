
import numpy as np
import pandas as pd

class linear_reg(object):
    """docstring for linear_reg."""
    def __init__(self, X, y):
        super(linear_reg, self).__init__()
        self.X = np.matrix(X)
        self.y = y.reshape(-1,1)
        self.learn(self.X, self.y)

    def learn(self, X, y, alpha=0.01, num_iters=10000):
        self.X = np.matrix(X)
        self.y = y.reshape(-1,1)
        # Rajouter 2 () apres np.zeros
        self.theta = np.zeros((np.size(self.X[0,:]) + 1, 1))

        def featureNormalize(X):
            mu = X.mean(axis = 0)
            sigma = X.std(axis = 0)
            X_norm = (X - mu) / sigma
            return X_norm, mu, sigma

        self.X, self.mean, self.stdev = featureNormalize(self.X)
        self.X = np.c_[np.ones(np.size(X[:,0])), self.X]

        def cost(X, y, theta):
            m = np.size(X[:,0])
            return (1 / (2 * m) * np.matmul((np.matmul(X,theta) - y).T, (np.matmul(X,theta) - y))).item(0)

        def gradientDescentMulti(X, y, theta, alpha, num_iters):
            m = np.size(X[:,0])
            j_history = []
            #Boucler sur le nombre d'itérations
            for i in range(num_iters):
                theta = theta - alpha / m * (np.matmul(X.T, (np.matmul(X,theta) - y)))
                j_history.append(cost(X, y, theta))
            return theta, j_history

        self.theta, self.j_history = gradientDescentMulti(self.X, self.y, self.theta, alpha, num_iters)

        return (self.theta)

    def get_theta(self):
        return (self.theta)

    def get_j_history(self):
        return (self.j_history)

    def predict(self, m):
        res = np.c_[np.ones(1), (m - self.mean) / self.stdev] * self.theta
        return (res)

if __name__ == "__main__":
    data = pd.read_csv("ex1data2.csv")
    X = np.c_[np.array(data['size'], float), np.array(data['nb_bedrooms'], float)]
    y = np.matrix([np.array(data['price'])], float)

    model = linear_reg(X, y)
    res = model.predict(np.matrix([1650, 3]));

    print("resultat = " + str(res.item(0)) + "$ pour 1650 feet^2 et 3 bedrooms\n avec coef : ")
    print(model.get_theta())
