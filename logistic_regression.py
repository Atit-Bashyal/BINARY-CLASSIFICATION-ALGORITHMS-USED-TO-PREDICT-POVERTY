#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:43:56 2019

@author: atit
"""

import os
import sys
import json

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from IPython.display import display
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression

# Function Spliting Training Data into Predictors, Labels and Weight dataframes 
def split_features_labels_weights(path,weights=['wt_ind', 'wt_hh'],weights_col=['wt_ind'],label_col=['poor']):
    data = pd.read_pickle(path)
    return (data.drop(weights + label_col, axis=1),
            data[label_col],
            data[weights_col])

TRAIN_PATH = "nepal_poverty_train.pkl"
QUESTION_PATH = "nepal_pov_questions.json"
TEST_PATH = "nepal_poverty_test.pkl"

#Function to load training data with columns standardized
def standardize(df, numeric_only=True):
    if numeric_only is True:
    # find non-boolean columns
        cols = df.loc[:, df.dtypes != 'uint8'].columns
    else:
        cols = df.columns
    for field in cols:
        mean, std = df[field].mean(), df[field].std()
        # account for constant columns
        if np.all(df[field] - mean != 0):
            df.loc[:, field] = (df[field] - mean) / std
    
    return df

def load_data(path, selected_columns=None, ravel=True, standardize_columns='numeric'):
    X, y, w = split_features_labels_weights(path)
    if selected_columns is not None:
        X = X[[col for col in X.columns.values if col in selected_columns]]
    if standardize_columns == 'numeric':
        standardize(X)
    elif standardize_columns == 'all':
        standardize(X, numeric_only=False)
    if ravel is True:
        y = np.ravel(y)
        w = np.ravel(w)
    return (X, y, w)
    
def get_coefs_df(X, coefs, index=None, sort=True):
    coefs_df = pd.DataFrame(np.std(X, 0) * coefs)
    coefs_df.columns = ["coef_std"]
    coefs_df['coef'] = coefs
    coefs_df['abs'] = coefs_df.coef_std.apply(abs)
    if index is not None:
        coefs_df.index = index
    if sort:
        coefs_df = coefs_df.sort_values('abs', ascending=False)
    return coefs_df



# load text of survey questions for reference
with open(QUESTION_PATH, 'r') as fp:
    questions = json.load(fp)

# load training data
X_train, y_train, w_train = load_data(TRAIN_PATH)

# summarize loaded data


# print first 5 rows of data
print(X_train.head())

X_train.dtypes


# Fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get an initial score
score = model.score(X_train, y_train)
print("In-sample score: {:0.2%}".format(score))
coefs = get_coefs_df(X_train, model.coef_[0])

# Load and transform the test set
X_test, y_test, w_test = load_data(TEST_PATH)

# Run the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Evaluate performance and store model


