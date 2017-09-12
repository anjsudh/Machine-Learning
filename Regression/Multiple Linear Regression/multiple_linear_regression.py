#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:56:20 2017

@author: anjanas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder_X = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder_X.fit_transform(X).toarray();

#Avoiding the dummy variable trap
X = X[:,1:]
#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )

#Fitting Multiple Linear Regression into our training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting test set results
Y_prediction = regressor.predict(X_test)

# Building optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones(shape = (50,1)).astype(int), values = X , axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
