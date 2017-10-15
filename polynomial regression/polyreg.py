# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


"""# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fit the data to model

# let us compare the linear and polynomial predictions

# taking linear first

#for linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
# returns
poly_reg.fit_transform(X)
#hence, save
X_poly = poly_reg.fit_transform(X)

#fit to dataset
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# plot the graphs
#for linear
plt.scatter(X, y , color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.show()

#for polynomial
plt.scatter(X, y ,color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.show()

#prediction
#linear
lin_reg.predict(5)

#poly
lin_reg_2.predict(poly_reg.fit_transform(5))

#vast difference