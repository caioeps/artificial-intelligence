import sys
import os
import re
from io import StringIO
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.dirname(__file__))

data_dir = "%s/dados_TC3/" % os.path.dirname(__file__)
data_file = "%s/aerogerador.dat" % data_dir
data_file = open(data_file, "r")

data = data_file.read()
data = re.sub('\t\n', '\n', data)
df = pd.read_csv(StringIO(data), delimiter="\t")
df.columns = ['winds', 'power']

winds = np.array(df['winds']).reshape(-1, 1)
power = np.array(df['power'])

# plt.plot(winds, power)
# plt.show()

n_iterations = 1

def test_mlp():
    P_success = []

    for i in range(0, n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(winds, power, test_size = 0.2, random_state = i)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(12,12,12,12), max_iter=100,
                            early_stopping=True, epsilon=0.00001)
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)

        n_success = [y_pred[i] for i in range(0, len(y_pred)) if y_pred[i] == y_test[i]]
        n_success = len(n_success)

        p_success = n_success / len(y_test) * 100
        P_success.append(p_success)
        print(P_success)

        pred = mlp.predict(winds)

        plt.figure('pred')
        plt.plot(winds, pred)
        plt.figure('actual')
        plt.plot(winds, power)
        plt.show()

    # print_stats(P_success)

def get_stats(P_success):
    p_min = min(P_success)
    p_max = max(P_success)
    p_mean = st.mean(P_success)
    p_std = st.stdev(P_success)
    return [p_min, p_max, p_mean, p_std]

def print_stats(P_success):
    p_min, p_max, p_mean, p_std = get_stats(P_success)
    print("p_min: ", p_min)
    print("p_max: ", p_max)
    print("p_mean: ", p_mean)
    print("p_std: ", p_std)

test_mlp()
