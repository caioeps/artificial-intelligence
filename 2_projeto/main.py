import sys
import os
import mimetypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))

from classifiers import MQClassifier, Perceptron
from preprocessing import load_data, get_images, get_subject_numbers

def mq_pca():
    print('MQ...')
    P_success = []
    for i in range(0, 20):
        data = load_data()
        subjects = get_subject_numbers(data)
        images = get_images(data)

        X_train, X_test, y_train, y_test = train_test_split(images, subjects, test_size = 0.2, random_state = i)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        pca = PCA(.95)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        mq = MQClassifier()
        mq.fit(X_train, y_train)

        y_pred = mq.predict(X_test)
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

def perceptron_pca():
    print('Perceptron...')
    P_success = []
    for i in range(0, 20):
        data = load_data()
        subjects = get_subject_numbers(data)
        images = get_images(data)

        X_train, X_test, y_train, y_test = train_test_split(images, subjects, test_size = 0.2, random_state = i)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        pca = PCA(.95)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)

        y_pred = perceptron.predict(X_test)
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

mq_pca()
perceptron_pca()

