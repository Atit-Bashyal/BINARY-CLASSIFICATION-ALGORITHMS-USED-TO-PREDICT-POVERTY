#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:46:37 2019

@author: atit
"""

import os
import sys
import json
import xlsxwriter 

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from IPython.display import display
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


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
                         return_values=True):
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



# Load the train and test set

X_train, y_train, w_train = load_data(TRAIN_PATH)
X_test, y_test, w_test = load_data(TEST_PATH)

###LDA with all features:

# Fit the model
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Get an initial score
score = model.score(X_train, y_train, w_train)
coefs = get_coefs_df(X_train, model.coef_[0])



# Run the model
y_pred_full = model.predict(X_test)
y_prob_full = model.predict_proba(X_test)[:,1]


#metrics and Prediction
metrics_lda_full = calculate_metrics(y_test,y_pred_full,y_prob_full,w_test)
pov_lda_full = predict_poverty_rate(TRAIN_PATH,TEST_PATH,model)

#results
conf_mat(metrics_lda_full)
metrics_table(metrics_lda_full,'lda_full')
pov_table(pov_lda_full,'lda_full')


#Transform LDA RESULTS
X_lda = model.transform(X_train)

mask = (y_train == 1)
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].scatter(X_lda[mask], y_train[mask], color='b', marker='+', label='poor')
axes[0].scatter(X_lda[~mask], y_train[~mask], color='r', marker='o', label='non-poor')
axes[0].set_title('LDA Projected Data')
axes[0].set_xlabel('Transformed axis')
axes[0].set_ylabel('\'Poor\' Probability')
axes[0].legend()

sns.kdeplot(np.ravel(X_lda[mask]), color='b', ax=axes[1], label='poor')
sns.kdeplot(np.ravel(X_lda[~mask]), color='r', ax=axes[1], label='non-poor')
axes[1].set_title('LDA Projected Density')
axes[1].set_xlabel('Transformed axis')
axes[1].set_ylabel('Class Density')
axes[1].legend()
plt.show()


### class Balance Oversampling

# Load and transform the training data
X_train, y_train, w_train = load_data(TRAIN_PATH)
cols = X_train.columns

# Apply oversampling with SMOTE
X_train, y_train = SMOTE().fit_sample(X_train, y_train)
print("X shape after oversampling: ", X_train.shape)

# Fit the model
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)


# Get an initial score
score = model.score(X_train, y_train)


# Store coefficients
coefs = get_coefs_df(X_train, model.coef_[0], index=cols)

# Load the test set
X_test, y_test, w_test = load_data(TEST_PATH)

# Run the model
y_pred_b = model.predict(X_test)
y_prob_b = model.predict_proba(X_test)[:,1]


metrics_lda_b = calculate_metrics(y_test,y_pred_b,y_prob_b,w_test)
pov_lda_b = predict_poverty_rate(TRAIN_PATH,TEST_PATH,model)


#results
conf_mat(metrics_lda_b)
metrics_table(metrics_lda_b,'lda_b')
pov_table(pov_lda_b,'lda_b')



#Transform LDA RESULTS
X_lda = model.transform(X_train)

mask = (y_train == 1)
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].scatter(X_lda[mask], y_train[mask], color='b', marker='+', label='poor')
axes[0].scatter(X_lda[~mask], y_train[~mask], color='r', marker='o', label='non-poor')
axes[0].set_title('LDA Projected Data')
axes[0].set_xlabel('Transformed axis')
axes[0].set_ylabel('\'Poor\' Probability')
axes[0].legend()

sns.kdeplot(np.ravel(X_lda[mask]), color='b', ax=axes[1], label='poor')
sns.kdeplot(np.ravel(X_lda[~mask]), color='r', ax=axes[1], label='non-poor')
axes[1].set_title('LDA Projected Density')
axes[1].set_xlabel('Transformed axis')
axes[1].set_ylabel('Class Density')
axes[1].legend()
plt.show()



###LDA with GRID SEARCH CV

# build the model
estimator = LinearDiscriminantAnalysis()
parameters = {'solver': ['svd']}

model = GridSearchCV(estimator, parameters, verbose=1, cv=5)
model.fit(X_train, y_train)

# Get an initial score
score = model.score(X_train, y_train)
coefs = get_coefs_df(X_train, model.best_estimator_.coef_[0])


# Run the model
y_pred_cv = model.predict(X_test)
y_prob_cv = model.predict_proba(X_test)[:,1]


metrics_lda_cv = calculate_metrics(y_test,y_pred_cv,y_prob_cv,w_test)
pov_lda_cv = predict_poverty_rate(TRAIN_PATH,TEST_PATH,model)


#results
conf_mat(metrics_lda_cv)
metrics_table(metrics_lda_cv,'lda_cv')
pov_table(pov_lda_cv,'lda_cv')



#Transform LDA RESULTS
X_lda = model.transform(X_train)

mask = (y_train == 1)
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].scatter(X_lda[mask], y_train[mask], color='b', marker='+', label='poor')
axes[0].scatter(X_lda[~mask], y_train[~mask], color='r', marker='o', label='non-poor')
axes[0].set_title('LDA Projected Data')
axes[0].set_xlabel('Transformed axis')
axes[0].set_ylabel('\'Poor\' Probability')
axes[0].legend()

sns.kdeplot(np.ravel(X_lda[mask]), color='b', ax=axes[1], label='poor')
sns.kdeplot(np.ravel(X_lda[~mask]), color='r', ax=axes[1], label='non-poor')
axes[1].set_title('LDA Projected Density')
axes[1].set_xlabel('Transformed axis')
axes[1].set_ylabel('Class Density')
axes[1].legend()
plt.show()



X_train_best_model = X_train


### Feature Selection with correlation using LDA axis
'''calculate the correlation between each feature column
 and the transformed axis that the LDA model produces.'''
 

def get_corrs_df(X, model):
    X_lda = model.transform(X)
    corrs = []
    for col in X.columns:
        corrs.append(np.corrcoef(np.ravel(X_lda),X[col])[0][1])
    corrs = pd.DataFrame(corrs, columns=['correlation'], index=X.columns)
    corrs['abs'] = corrs.correlation.apply(abs)
    return corrs.sort_values('abs', ascending=False)

X_train, y_train, w_train = load_data(TRAIN_PATH)
X_test, y_test, w_test = load_data(TEST_PATH)


corrs = get_corrs_df(X_train, model.best_estimator_)
display(corrs.head(20))
feats = corrs.index.values

'''we eleminate columns with low coorelation since they do not impact 
the transformed LDA axis'''

feats = corrs[corrs['abs'] > 0.25].index.values
print(feats)

# Load and transform the new training data
X_train, y_train, w_train = load_data(TRAIN_PATH, selected_columns=feats)
cols = X_train.columns

# Load and transform the new test set
X_test, y_test, w_test = load_data(TEST_PATH, selected_columns=feats)

# Apply oversampling with SMOTE
X_train, y_train = SMOTE().fit_sample(X_train, y_train)
X_train = pd.DataFrame(X_train, columns=cols)


# build the model
estimator = LinearDiscriminantAnalysis()
parameters = {'solver': ['svd', 'lsqr']}

model = GridSearchCV(estimator, parameters, verbose=1, cv=5)
model.fit(X_train, y_train)

# Get an initial score
score = model.score(X_train, y_train)
print("In-sample score: {:0.2%}".format(score))
coefs = get_coefs_df(X_train, model.best_estimator_.coef_[0])
print("Best model parameters:", model.best_params_)

# Run the model
y_pred_feats_cv = model.predict(X_test)
y_prob_feats_cv = model.predict_proba(X_test)[:,1]

metrics_feats = calculate_metrics(y_test, y_pred_feats_cv, y_prob_feats_cv, w_test)



#results
conf_mat(metrics_feats)
metrics_table(metrics_feats,'lda_feats')




X_lda = model.transform(X_train)
mask = (y_train == 1)
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].scatter(X_lda[mask], y_train[mask], color='b', marker='+', label='poor')
axes[0].scatter(X_lda[~mask], y_train[~mask], color='r', marker='o', label='non-poor')
axes[0].set_title('LDA Projected Data')
axes[0].set_xlabel('Transformed axis')
axes[0].set_ylabel('\'Poor\' Probability')
axes[0].legend()

sns.kdeplot(np.ravel(X_lda[mask]), color='b', ax=axes[1], label='poor')
sns.kdeplot(np.ravel(X_lda[~mask]), color='r', ax=axes[1], label='non-poor')
axes[1].set_title('LDA Projected Density')
axes[1].set_xlabel('Transformed axis')
axes[1].set_ylabel('Class Density')
axes[1].legend()
plt.show()







    





    
