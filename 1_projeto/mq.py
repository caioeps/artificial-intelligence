#!/usr/bin/env python3

import pandas as pd
import numpy as np
import statistics as st
from sklearn import model_selection

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

data = pd.read_csv('./lung-cancer.data', sep = ',', header = None)

# Filter corrupted attributes.
ignored_columns = [4, 38]
data.drop(data.columns[ignored_columns], axis=1, inplace=True)

X = np.array(data.drop(data.iloc[:,-1] , axis = 1)) # Variáveis independentes
y = np.array(data.iloc[:, -1]) # escolhendo a variável dependente, ultima coluna

P_success = []

for i in range(0, 100):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = i)

    # testando o modelo
    model = MQClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = [int(round(x)) for x in y_pred]

    n_success = [y_pred[i] for i in range(0, len(y_pred)) if y_pred[i] == y_test[i]]
    n_success = len(n_success)

    p_success = n_success / len(y_test) * 100
    P_success.append(p_success)

P_success_min = min(P_success)
P_success_max = max(P_success)
P_success_mean = st.mean(P_success)
P_success_std = st.stdev(P_success)

print("P_success_min: ", P_success_min)
print("P_success_mean: ", P_success_mean)
print("P_success_std: ", P_success_std)

