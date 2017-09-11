#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 23:26:08 2017

@author: anjanas
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )

#Fitting the Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()     
regressor.fit(X_train, Y_train)

#Predicting the Test set results
Y_pred = regressor.predict(X_test)

#Plotting graph
plt.scatter(X_train, Y_train, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary vs Experience - Training Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, c='red')
plt.plot(X_test, regressor.predict(X_test), c='blue')
plt.title('Salary vs Experience - Prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()