import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

class MQClassifier(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        # adiciona coluna de 1 nos dados
        X_train = np.insert(X_train, 0, 1, 1)

        # (X^T * X)^-1 * X^T * y
        # w = y_train @ X.T @ np.linalg.pinv(X @ X.T)
        # w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X)),X_treino.T),y_train)
        w = np.dot(np.linalg.pinv(X_train), y_train)

        self.w = w
        self.intercept = self.w[0] # w0
        self.coef = self.w[1:] # .. + x11*w1 + x12*w2 + ...

    def predict(self, X):
        # adiciona coluna de 1 nos dados
        X = np.insert(X, 0, 1, 1)
        return np.dot(X, self.w) # Y = X * W
