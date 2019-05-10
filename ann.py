# -*- coding: utf-8 -*-
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X_one_hot_encoded = one_hot_encoder.fit_transform(X).toarray()
X_one_hot_encoded = X_one_hot_encoded[:, 1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_one_hot_encoded, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, activation='relu', input_dim=11, kernel_initializer='uniform'))
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

(cm[0][0] + cm[1][1]) / 2000

# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000

import numpy as np

new_pred = classifier.predict(sc.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)