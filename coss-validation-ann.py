# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder

first_labelEncoder = LabelEncoder()
X_encoded = X
X_encoded[:, 1] = first_labelEncoder.fit_transform(X[:, 1])
second_labelEncoder = LabelEncoder()
X_encoded[:, 2] = second_labelEncoder.fit_transform(X[:, 2])

X_oneHotEncoded = pd.get_dummies(X_encoded[:, 1], drop_first=True).values

X_preprocessed = np.zeros((X_encoded.shape[0], X_encoded.shape[1] + X_oneHotEncoded.shape[1] - 1))
X_preprocessed[:, 0] = X_encoded[:, 0]
X_preprocessed[:, 1:3] = X_oneHotEncoded[:, :]
X_preprocessed[:, 3:] = X_encoded[:, 2:]

from keras.models import Sequential
from keras.layers import Dense


def build_model():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

classifier = KerasClassifier(build_fn=build_model, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X=X_preprocessed, y=Y, cv=10, n_jobs=-1)
accuracy_mean = accuracies.mean()
accuracy_variance = accuracies.std()
