#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:15:53 2019

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

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV


from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    roc_auc_score,
    accuracy_score,
    precision_score
)

from sklearn.metrics import (
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_curve,
    auc
)

TRAIN_PATH = "nepal_poverty_train.pkl"
QUESTION_PATH = "nepal_pov_questions.json"
TEST_PATH = "nepal_poverty_test.pkl" 


ALGORITHM_NAME = 'lda'
COUNTRY = 'Nepal'

with open(QUESTION_PATH, 'r') as fp:
    questions = json.load(fp)
    
def split_features_labels_weights(path,weights=['wt_ind', 'wt_hh'],weights_col=['wt_ind'],label_col=['poor']):
    data = pd.read_pickle(path)
    return (data.drop(weights + label_col, axis=1),
            data[label_col],
            data[weights_col])
    
''' standardize the features (substracting mean from the columns)'''

def standardize(df, numeric_only=True):
    if numeric_only is True:
    # find non-boolean columns
        cols = df.loc[:, df.dtypes != 'uint8'].columns
    else:
        cols = df.columns
    for field in cols:
        m, s = df[field].mean(), df[field].std()
        # account for constant columns
        if np.all(df[field] - m != 0):
            df.loc[:, field] = (df[field] - m) / s
    
    return df

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

def clip_yprob(y_prob):
    """Clip yprob to avoid 0 or 1 values. Fixes bug in log_loss calculation
    that results in returning nan."""
    eps = 1e-15
    y_prob = np.array([x if x <= 1-eps else 1-eps for x in y_prob])
    y_prob = np.array([x if x >= eps else eps for x in y_prob])
    return y_prob

def predict_poverty_rate(train_path, test_path, model,
                         standardize_columns='numeric',
                         ravel=True,
                         selected_columns=None,
                         show=True,
                         return_values=False):
    # Recombine the entire dataset to get the actual poverty rate
    X_train, y_train, w_train = load_data(TRAIN_PATH,
                                          standardize_columns=standardize_columns,
                                          ravel=ravel,
                                          selected_columns=selected_columns)
    X_test, y_test, w_test = load_data(TEST_PATH,
                                       standardize_columns=standardize_columns,
                                       ravel=ravel,
                                       selected_columns=selected_columns)
    pov_rate = pd.DataFrame(np.vstack((np.vstack((y_train, w_train)).T,
                                       np.vstack((y_test, w_test)).T)),
                            columns=['poor', 'wta_pop'])
    pov_rate_actual = (pov_rate.wta_pop * pov_rate.poor).sum() / pov_rate.wta_pop.sum()

    # Make predictions on entire dataset to get the predicted poverty rate
    pov_rate['pred'] = model.predict(np.concatenate((X_train.as_matrix(), X_test.as_matrix())))
    pov_rate_pred = (pov_rate.wta_pop * pov_rate.pred).sum() / pov_rate.wta_pop.sum()

    if show == True:
        print("Actual poverty rate: {:0.2%} ".format(pov_rate_actual))
        print("Predicted poverty rate: {:0.2%} ".format(pov_rate_pred))
    if return_values:
        return pov_rate_actual, pov_rate_pred
    else:
        return
    
def calculate_metrics(y_test, y_pred, y_prob=None, sample_weights=None):
    """Cacluate model performance metrics"""

    # Dictionary of metrics to calculate
    metrics = {}
    metrics['confusion_matrix']  = confusion_matrix(y_test, y_pred, sample_weight=sample_weights)
    metrics['roc_auc']           = None
    metrics['accuracy']          = accuracy_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['precision']         = precision_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['recall']            = recall_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['f1']                = f1_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['cohen_kappa']       = cohen_kappa_score(y_test, y_pred)
    metrics['cross_entropy']     = None
    metrics['fpr']               = None
    metrics['tpr']               = None
    metrics['auc']               = None

    # Populate metrics that require y_prob
    if y_prob is not None:
        clip_yprob(y_prob)
        metrics['cross_entropy']     = log_loss(y_test,
                                                clip_yprob(y_prob), 
                                                sample_weight=sample_weights)
        metrics['roc_auc']           = roc_auc_score(y_test,
                                                     y_prob, 
                                                     sample_weight=sample_weights)

        fpr, tpr, _ = roc_curve(y_test,
                                y_prob, 
                                sample_weight=sample_weights)
        metrics['fpr']               = fpr
        metrics['tpr']               = tpr
        metrics['auc']               = auc(fpr, tpr, reorder=True)

    return metrics

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

SUBSAMPLE = 1

def subsample(X, y, w, stratify=True, seed=566):
    n_samples = int(SUBSAMPLE * X.shape[0])
    
    rng = np.random.RandomState(seed)
    
    if stratify:
        y_rate = y.mean()
        n_true = int(n_samples * y_rate)
        n_false = n_samples - n_true
        
        true_idx = rng.choice(np.where(y)[0], n_true, replace=False)
        false_idx = rng.choice(np.where(~y)[0], n_false, replace=False)
        
        sample_idx = np.union1d(true_idx, false_idx)
    else:
        sample_idx = rng.choice(np.arange(X.shape[0]), n_samples, replace=False)
    
    return X.iloc[sample_idx, :], y[sample_idx], w[sample_idx]

### KNN with all Features and n=15
    
X_train, y_train, w_train = load_data(TRAIN_PATH)
X_train, y_train, w_train = subsample(X_train, y_train, w_train)


# Fit the model
model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_train, y_train)

# Get an initial score
score = model.score(X_train, y_train, w_train)
print("In-sample score: {:0.2%}".format(score))

# Load the test set
X_test, y_test, w_test = load_data(TEST_PATH)

# Run the model
y_pred_knn15 = model.predict(X_test)
y_prob_knn15 = model.predict_proba(X_test)[:,1]

metrics_knn15 = calculate_metrics(y_test,y_pred_knn15,y_prob_knn15,w_test)
print(metrics_knn15)



### 5 fold CV with full features


# build the model
estimator = KNeighborsClassifier()
parameters = {'n_neighbors': [3,5,7,9,11,13,15]}


model = GridSearchCV(estimator, parameters, verbose=4, cv=5)
model.fit(X_train, y_train)

# Get an initial score
score = model.score(X_train, y_train)
print("In-sample score: {:0.2%}".format(score))
print("Best model parameters:", model.best_params_)

# Load the test set
X_train, y_train, w_train = load_data(TRAIN_PATH)
cols = X_train.columns
X_train.shape

# Apply oversampling with SMOTE
X_train, y_train = SMOTE().fit_sample(X_train, y_train)
print("X shape after oversampling: ", X_train.shape)



# Run the model
y_pred_knncv = model.predict(X_test)
y_prob_knncv = model.predict_proba(X_test)[:,1]

pred_ov =  predict_poverty_rate(TRAIN_PATH,TEST_PATH,model)
metrics_knncv = calculate_metrics(y_test,y_pred_knncv,y_prob_knncv,w_test)
print(metrics_knncv)




