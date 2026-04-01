#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:01:12 2026

@author: nikita
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from GPin_GBTSVM import GPinGTSVM
from gen_ball import gen_balls


# ---------------- Data Loader ---------------- #
def load_dataset(file_path):
    df = pd.read_csv(file_path, header=None)
    X = df.values.astype(float)

    # Convert labels (0 → -1)
    X[X[:, -1] == 0, -1] = -1

    y = X[:, -1].reshape(-1, 1)
    X = X[:, :-1]

    return X, y


# ---------------- Training + Evaluation ---------------- #
def run_model(file_path):
    X, y = load_dataset(file_path)

    # Fixed parameters (modify if needed)
    C1, C2 = 100, 1
    tau1, tau2 = 0.5, 0.25
    eps1, eps2 = 0.1, 0.1

    kf = KFold(n_splits=10, shuffle=True, random_state=10)

    accuracy_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if len(np.unique(y_test)) == 1:
            continue

        # -------- Granular Ball Generation -------- #
        train_data = np.hstack((X_train, y_train))
        balls = gen_balls(train_data, pur=1, delbals=4)

        centers = np.array([b[0] for b in balls])
        radius = np.array([b[1] for b in balls]).reshape(-1, 1)
        labels = np.array([b[-1] for b in balls]).reshape(-1, 1)

        X_train_new = np.hstack((centers, radius, labels))

        # -------- Model -------- #
        model = GPinGTSVM(C1, C2, C1, C2, tau1, tau2, tau1, tau2, eps1, eps2, eps1, eps2)
        model.fit(X_train_new)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracy_list.append(acc * 100)

    return np.mean(accuracy_list)


# ---------------- Example Usage ---------------- #  
if __name__ == "__main__":
    file_path = "your_dataset.csv"   # replace with your dataset
    accuracy = run_model(file_path)

    print(f"\nAccuracy: {accuracy:.2f}%")