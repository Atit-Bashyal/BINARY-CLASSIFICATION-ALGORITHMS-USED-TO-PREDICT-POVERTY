#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:25:19 2019

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
from sklearn.linear_model import LogisticRegressionCV
from models import evaluate_model
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import seaborn as sn


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

import xlsxwriter




TRAIN_PATH = "nepal_poverty_train.pkl"
QUESTION_PATH = "nepal_indv_questions.json"
TEST_PATH = "nepal_poverty_test.pkl" 



with open(QUESTION_PATH, 'r') as fp:
    questions = json.load(fp)
    
    
ALGORITHM_NAME = LogisticRegression()
COUNTRY = "Nepal"

'''defining a function to split data into features, label variable and the weight'''



def clip_yprob(y_prob):
    """Clip yprob to avoid 0 or 1 values. Fixes bug in log_loss calculation
    that results in returning nan."""
    eps = 1e-15
    y_prob = np.array([x if x <= 1-eps else 1-eps for x in y_prob])
    y_prob = np.array([x if x >= eps else eps for x in y_prob])
    return y_prob

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



def get_vif(X):
    vi_factors = [variance_inflation_factor(X.values, i)
                             for i in range(X.shape[1])]
    
    return pd.Series(vi_factors,
                     index=X.columns,
                     name='variance_inflaction_factor')


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

def predict_poverty_rate(train_path, test_path, model,
                         standardize_columns='numeric',
                         ravel=True,
                         selected_columns=None,
                         show=True,
                         return_values= True):
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

def conf_mat(metrics):
    array = metrics['confusion_matrix']
    classes=['poor','non-poor']
    df_cm = pd.DataFrame(array, index = [i for i in classes],
                  columns = [i for i in classes])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,cmap="Blues")

def metrics_table(metrics,model_name):
    workbook = xlsxwriter.Workbook(model_name+'.xlsx') 
    worksheet = workbook.add_worksheet() 
    metrics.pop('confusion_matrix')
    metrics.pop('fpr')
    metrics.pop('tpr')
    metrics.pop('cohen_kappa')
    metrics.pop('cross_entropy')
    metrics.pop('roc_auc')

    head = []
    for key in metrics:
        head.append(key)
    row=0
    column = 0
    for item in head:
        worksheet.write(row,column,item)
        column +=1
    row = 1
    column = 0
    for key in metrics:
        worksheet.write(row, column, metrics[key]) 
        column +=1 
        
    workbook.close()
    
def pov_table(pred,model_name):
    workbook = xlsxwriter.Workbook(model_name+'pov.xlsx') 
    worksheet = workbook.add_worksheet() 
    head = ['Actual','Predicted']
    row=0
    column = 0
    for item in head:
        worksheet.write(row,column,item)
        column +=1
    row = 1
    column = 0
    for i in range(len(head)):
        worksheet.write(row, column, pred[i]) 
        column +=1 
        
    workbook.close()

    
# Load and transform the training data
X_train, y_train, w_train = load_data(TRAIN_PATH)
# Load and transform the test set
X_test, y_test, w_test = load_data(TEST_PATH)



# Fit the model without weights
model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_train, y_train)

# Run the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] 

#Prediction and Metrics
pred = predict_poverty_rate(TRAIN_PATH,TEST_PATH,model,return_values=True)
metrics_lr = calculate_metrics (y_test, y_pred, y_prob)

##results
conf_mat(metrics_lr)
metrics_table(metrics_lr,'logisticreg')
pov_table(pred,'logisticreg')



#### model with weights
model.fit(X_train, y_train, sample_weight=w_train)
score_w = model.score(X_train, y_train, sample_weight=w_train)
coefs_w = get_coefs_df(X_train, model.coef_[0])['abs']

# Run the model
y_pred_w = model.predict(X_test)
y_prob_w = model.predict_proba(X_test)[:,1]

# prediction and Metrics
pred_w = predict_poverty_rate(TRAIN_PATH,TEST_PATH,model)
metrics_lr_w = calculate_metrics (y_test, y_pred_w, y_prob_w)

##results
conf_mat(metrics_lr_w)
metrics_table(metrics_lr_w,'logisticreg_weights')
pov_table(pred_w,'logisticreg_weights')


### model with cross validation

# Load and transform the training data
X_train, y_train, w_train = load_data(TRAIN_PATH)

# Fit the model
model = LogisticRegressionCV(Cs=10, cv=5, verbose=1)
model.fit(X_train, y_train, sample_weight=w_train)

# Get an initial score
score = model.score(X_train, y_train, sample_weight=w_train)

coefs = get_coefs_df(X_train, model.coef_[0])

# Display best parameters
best_params = model.C_[0]

# Run the model
y_pred_cv = model.predict(X_test)
y_prob_cv = model.predict_proba(X_test)[:,1]


#prediction and Metrics
pred_cv = predict_poverty_rate(TRAIN_PATH,TEST_PATH,model)
best_model_metrics= calculate_metrics (y_test, y_pred_cv, y_prob_cv, w_test)


##results
conf_mat(best_model_metrics)
metrics_table(best_model_metrics,'logisticreg_weights')
pov_table(pred_cv,'logisticreg_weights')



### feature selection using l1 reqularization 

# Fit the model
model = LogisticRegressionCV(cv=5, penalty='l1', Cs=[3e-5] , solver='liblinear')
model.fit(X_train, y_train, sample_weight=w_train)
coefs = get_coefs_df(X_train, model.coef_[0])
coefs = coefs[coefs.coef != 0]
print("{} features selected".format(coefs.shape[0]))
display(coefs)
feats_l1 = coefs.index.values
print(feats_l1)

# model with fetures selected by l1 regularization
# Load and transform the training data
X_train, y_train, w_train = load_data(TRAIN_PATH, selected_columns=feats_l1)

# Fit the model
model = LogisticRegressionCV(Cs=10, cv=5)
model.fit(X_train, y_train, sample_weight=w_train)

# Get an initial score
score = model.score(X_train, y_train, sample_weight=w_train)
print("In-sample score: {:0.2%}".format(score))
coefs = get_coefs_df(X_train, model.coef_[0])

# Display best parameters
print("Best model parameters: C={}".format(model.C_[0]))

# Load the test set
X_test, y_test, w_test = load_data(TEST_PATH, selected_columns=feats_l1)

# Run the model
y_pred_l1 = model.predict(X_test)
y_prob_l1 = model.predict_proba(X_test[feats_l1])[:,1]

#prediction and Metrics
best_model_metrics_l1= calculate_metrics (y_test, y_pred_l1, y_prob_l1, w_test)


##results
conf_mat(best_model_metrics_l1)
metrics_table(best_model_metrics_l1,'logisticreg_weights')





###class balance
# Load and transform the training data
X_train, y_train, w_train = load_data(TRAIN_PATH)
cols = X_train.columns
X_train.shape

# Apply oversampling with SMOTE
X_train, y_train = SMOTE().fit_sample(X_train, y_train)
print("X shape after oversampling: ", X_train.shape)

# Fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get an initial score
score = model.score(X_train, y_train)
print("In-sample score: {:0.2%}".format(score))

# Store coefficients
coefs = get_coefs_df(X_train, model.coef_[0], index=cols)

# Load the test set
X_test, y_test, w_test = load_data(TEST_PATH)

# Run the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

metrics_ov= calculate_metrics (y_test, y_pred, y_prob, w_test)
pred_ov =  predict_poverty_rate(TRAIN_PATH,TEST_PATH,model)

##results
conf_mat(metrics_ov)
metrics_table(metrics_ov,'logisticreg_weights')
pov_table(pred_ov,'logisticreg_weights')
