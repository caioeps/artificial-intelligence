import random
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data from CSV.
dataset = pd.read_csv("./lung-cancer.data", header=None)

# Filter corrupted attributes.
ignored_columns = [4, 38]
dataset.drop(dataset.columns[ignored_columns], axis=1, inplace=True)

# Labels
Y = dataset.iloc[:, 0].values
Y_new = []

# Represent labels as matrices.
for y in Y:
    y_new = [0,0,0]
    y_new[y-1] = 1
    Y_new.append(y_new)
Y = np.array(Y_new)

# Remove labels from dataset. It can probably be done with iloc.
dataset.drop(dataset.columns[0], axis=1, inplace=True)

X = dataset.iloc[:, :].values
X = np.array(X)

P_fail = []
P_success = []

for i in range(0, 100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = i)

    # Normalize train and test variables.
    scaler  = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test  = scaler.transform(X_test)

    # In order to calculate W, we need to transpose them all.
    # CxL -> LxC
    X_train = X_train.transpose()
    X_test  = X_test.transpose()
    Y_train = Y_train.transpose()
    Y_test  = Y_test.transpose()

    # Calculate the weight matrix.
    W = (Y_train @ X_train.transpose()) @ np.linalg.pinv((X_train @ X_train.transpose()) + (random.uniform(0, 1) * np.eye(len(X_test))))

    # Calculate test data.
    Y_pred = W @ X_test

    # print(Y_test.shape)
    # print(Y_pred.shape)
    # print(Y_test)
    # print(Y_pred)

    # Y_pred = scaler.inverse_transform(Y_pred.transpose())

    Y_pred_t = Y_pred.transpose()
    Y_test_t = Y_test.transpose()

    # [0.1, 0.89, 0.80] => [0, 1, 0]
    for i in range(0, len(Y_pred_t) - 1):
        y_pred = Y_pred_t[i]
        Y_pred_t[i] = [1 if x == max(y_pred) else 0 for x in y_pred]

    n_fails = 0
    n_success = 0
    for i in range(0, len(Y_pred_t) - 1):
        y_pred = Y_pred_t[i]
        y_test = Y_test_t[i]

        if y_pred == Y_test:
            n_success += 1
        else:
            n_fails += 1

    P_success.append(100 * (n_success / len(Y_pred_t)))
    P_fail.append(100 * (n_fails / len(Y_pred_t)))

P_fail_min = min(P_fail)
P_fail_max = max(P_fail)
P_fail_mean = st.mean(P_fail)
P_fail_std = st.stdev(P_fail)

print("P_fail_min: ", P_fail_min)
print("P_fail_mean: ", P_fail_mean)
print("P_fail_std: ", P_fail_std)
