# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imp = imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder_X = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder_X.fit_transform(X).toarray();

from sklearn.preprocessing import LabelEncoder
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )

         