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

from classifiers import MQClassifier, Perceptron, MLPClassifier
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

    print_stats(P_success)

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

    print_stats(P_success)

def mlp_pca():
    print('MLP...')
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

        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        y_pred = [int(round(x)) for x in y_pred]
        n_success = [y_pred[i] for i in range(0, len(y_pred)) if y_pred[i] == y_test[i]]
        n_success = len(n_success)

        p_success = n_success / len(y_test) * 100
        P_success.append(p_success)

    print_stats(P_success)

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

mq_pca()
perceptron_pca()
mlp_pca()

