#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:19:09 2017

@author: vyomunadkat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:54:12 2017

@author: vyomunadkat
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = np.reshape(X_train, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))
X_test = np.reshape(X_test, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))
regressor.fit(X_train, y_train)

# predict the dependent data
y_pred = regressor.predict(X_test)

# plotting the training set graph
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('R&D v/s Profit (Training)')
plt.xlabel('Experience')
plt.ylabel('Profit')
plt.show()

# plotting the test set graph
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('R&D v/s Profit (Test)')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()


# plotting points of actual test and predicted test￼
plt.scatter(X_test, y_test, color = 'blue')
plt.scatter(X_test, y_pred, color = 'red')
plt.title('R&D v/s Profit (Test & Prediction)')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()

