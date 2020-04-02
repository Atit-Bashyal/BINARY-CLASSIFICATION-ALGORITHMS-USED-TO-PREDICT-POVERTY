#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:57:38 2019

@author: atit
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader

from matplotlib import pyplot as plt
from IPython.display import display
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split



filepath_household = "nlss_3.dta"
filepath_individual = "poverty_nlss.dta"

'''function for loading stata file'''

def load_stata(filepath,indexcol, drop_minornans=False):
    
    data = pd.read_stata(filepath, convert_categoricals=False).set_index(indexcol)


# convert the categorical variables into
# the category type
    with StataReader(filepath) as reader:
        reader.value_labels()
    
 
        mapping = {col: reader.value_label_dict[t] for col, t in 
                   zip(reader.varlist, reader.lbllist)
                   if t in reader.value_label_dict}
       
    
    
# drop records with name labels.    
        
        data.replace(mapping, inplace=True)
       # convert the categorical variables into
        # the category type
        cat_list = []
        for c in data.columns:
            if c in mapping:
                cat_list.append(c)
                data[c] = data[c].astype('category')
        data['poor'] = data['poor'].astype('category')
        data.drop('gap',axis=1,inplace=True)
        data.drop('gapsq',axis=1,inplace=True)
        data.drop('food_poor',axis=1,inplace=True)
        data.drop('inc_poor',axis=1,inplace=True)
      
        data.drop('Date',axis=1,inplace=True)
       
        
        for i in data.columns:
            if data[i].dtype == "object":
                data.drop(i, axis=1, inplace=True)
        # drop records with only a few nans
        
        if drop_minornans: 
            nan_counts = (data.applymap(pd.isnull)
                          .sum(axis=0)
                          .sort_values(ascending=False))
            nan_cols = nan_counts[(nan_counts > 0) & (nan_counts < 10)].index.values
            data = data.dropna(subset=nan_cols)
      
                
        questions = reader.variable_labels()
        
        
    
        
    return data, questions, cat_list


'''load data'''
    

nep_hhold , nep_hhold_q, c_list = load_stata(filepath_household,['xhnum','xhpsu'],drop_minornans=True)
print(nep_hhold.head())                                             
nep_hhold.columns[1:5]
pd.DataFrame.from_dict(nep_hhold_q, orient='index')[0:10]
print (c_list)
s = 'Nepal household data has {:,} rows and {:,} columns'
print(s.format(nep_hhold.shape[0], nep_hhold.shape[1]))
s = 'Percent poor: {:0.1%} \tPercent non-poor: {:0.1%}'
per = nep_hhold.poor.value_counts(normalize=True)
print(s.format(per[1],per[0]))
print(nep_hhold.head())

print(nep_hhold_q)
nep_hhold.dtypes[0:50]
nep_hhold.Eastern
pd.DataFrame.from_dict(nep_hhold_q, orient='index')[40:100]

g = sns.catplot(y='poor',kind="count", orient="h",palette="pastel", data=nep_hhold)

#Demographics





#Expoloring Education
#highest level of household head education
#only look at interesting feature in 1
#answered yes to below the literaciy rate:
sns.catplot(x="poor", hue="head_educ_1", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)
# Answered true to question literate or below five years of primary education:
sns.catplot(x="poor", hue="head_educ_2", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)
# answered yes to completed 7 years of education
sns.catplot(x="head_educ_5", hue="poor", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)
#answered yes to completed highschool
sns.catplot(x="head_educ_5", hue="poor", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)
print(nep_hhold.head_edu)

# minority groups
#muslim
sns.catplot(x="poor", hue="head_ethgroup_10", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)
#Janajati Hill
sns.catplot(x="poor", hue="head_ethgroup_8", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)
#Janajati Terai
sns.catplot(x="poor", hue="poor", kind="head_ethgroup",palette="pastel", edgecolor=".6",data=nep_hhold)


sns.catplot(y="head_ethgroup", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)
sns.catplot(x="wi30_hosp", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold,)
sns.catplot(y="max_educ_fem", kind="count",palette="pastel", edgecolor=".6",data=nep_hhold)



sns.distplot(nep_hhold.head_age, hist= False)

sns.distplot(nep_hhold.hhsize,)





#load the individual level data
nep_indv, nep_indv_q , c_idv_list= load_stata(filepath_individual,['xhnum','v07_idc'])

'''s = 'Nepal household data has {:,} rows and {:,} columns'
print(s.format(nep_indv.shape[0], nep_indv.shape[1]))
s = 'Percent poor: {:0.1%} \tPercent non-poor: {:0.1%}'
per = nep_indv.poor.value_counts(normalize=True)
print(s.format(per[1],per[0]))
print(nep_indv.head())'''


###changing target variable poor into boolean
nep_hhold.poor = (nep_hhold.poor == 1)

nep_hhold.wt_hh


'''Derive some features from individual data if needed'''


#looking into features in individual data:

'''pd.DataFrame.from_dict(nep_indv_q, orient='index')[150:200]'''

'''Feature Inspection and Reduction'''

nep_hhold = pd.get_dummies(nep_hhold, drop_first=True, dummy_na=True, prefix_sep='__' )
print("Nepal household shape with dummy variables added", nep_hhold.shape)

# remove columns with only one unique value removing survey answers that remain same
nep_hhold = nep_hhold.loc[:, nep_hhold.nunique(axis=0) > 1]
print("Nepal household shape with constant columns dropped", nep_hhold.shape)

# remove duplicate columns - questions with identical ansewrs for each household
def drop_duplicate_columns(df, ignore=[], inplace=False):
    if not inplace:
        df = df.copy()

    # pairwise correlations
    corr = df.corr()
    corr[corr.columns] = np.triu(corr, k=1)
    corr = corr.stack()

    # for any perfectly correlated variables, drop one of them
    for ix, r in corr[(corr == 1)].to_frame().iterrows():
        first, second = ix

        if second in df.columns and second not in ignore:
            df.drop(second, inplace=True, axis=1)

    if not inplace:
        return df
    
drop_duplicate_columns(nep_hhold, ignore=['wt_ind', 'wt_hh'], inplace=True)
print("Nepal household shape with duplicate columns dropped", nep_hhold.shape)

### Drop rows wint missing values and NA
nep_hhold = nep_hhold.dropna()
print("Nepal household shape with  NA rows dropped", nep_hhold.shape)

#Explore Data


def plot_numeric_hist(df, 
                      col, 
                      x_label, 
                      y_label='Percentage of Households', 
                      target='poor', 
                      integer_ticks=True, 
                      ax=None):
    if ax is None:
        ax = plt.gca()
    
    df.groupby(df[target])[col].plot.hist(bins=np.arange(0, df[col].max()) - 1, 
                                          alpha=0.5, 
                                          normed=True, 
                                          ax=ax)

    ax.set_xlim([0,df[col].max()])
    if integer_ticks:
        ax.set_xticks(np.arange(0,df[col].max()) + 0.5)
        ax.set_xticklabels(np.arange(0,df[col].max()+1, dtype=int))
        ax.xaxis.grid(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title='poor')
pd.DataFrame.from_dict(nep_hhold_q, orient='index')[200:260]

#Distribution of childeren
nep_hhold['nchild'] = nep_hhold.nkid06 + nep_hhold.nkid715
plot_numeric_hist(nep_hhold,'nchild','Number of children')



#Distribution of adults
nep_hhold['nadult'] = nep_hhold.nadultm + nep_hhold.nadultf
plot_numeric_hist(nep_hhold,'nadult','Number of Adult')

#employment
plot_numeric_hist(nep_hhold,'nemp','Number of employed adults')
plot_numeric_hist(nep_hhold,'nemp_male','Number of employed adult male')
plot_numeric_hist(nep_hhold,'nemp_fem','Number of employed adult female')
sns.catplot(x="head_occup_2", kind="bar",hue='poor', edgecolor=".6",data=nep_hhold)


#Migration
sns.catplot(x="has_formig", kind="box",hue='poor',data=nep_hhold)

#Education












#2d distributions
sns.jointplot(x='', y='hhsize',kind='kde',color='k', data=nep_hhold)
sns.kdeplot(nep_hhold.nemp_male, nep_hhold.nemp_fem)






#share of childeren vs household size
nep_hhold['child'] = nep_hhold.s_kid06 + nep_hhold.s_kid715
sns.catplot(x="hhsizeT",y='child', hue='poor',palette="pastel", kind='point',markers=["^", "o"], linestyles=["-", "--"],data=nep_hhold) 
sns.catplot(x="hhsize",y='child', hue='poor',palette="pastel", kind='point',markers=["^", "o"], linestyles=["-", "--"],data=nep_hhold)
 
#share of adults VS household size
nep_hhold['adult'] = nep_hhold.s_adultm + nep_hhold.s_adultf
sns.catplot(x="hhsizeT",y='adult', hue='poor',palette="pastel", kind='point',markers=["^", "o"], linestyles=["-", "--"],data=nep_hhold) 
sns.catplot(x="hhsize",y='adult', hue='poor',palette="pastel", kind='point',markers=["^", "o"], linestyles=["-", "--"],data=nep_hhold)
#share of elders VS household size
sns.catplot(x="hhsizeT",y='s_elderly', hue='poor',palette="pastel", kind='point',markers=["^", "o"], linestyles=["-", "--"],data=nep_hhold) 
sns.catplot(x="hhsize",y='s_elderly', hue='poor',palette="pastel", kind='point',markers=["^", "o"], linestyles=["-", "--"],data=nep_hhold)
 




plot_numeric_hist(nep_hhold,'hhsizeT','Capped Household size')


    
# Compare poor vs non-poor numeric features
#   We have 8 numeric features, so we make a 2x3 grid to plot them
fig, axes = plt.subplots(2, 4, figsize=(20,10))
plot_numeric_hist(nep_hhold, 
                  'hhsize', 
                  'Household Size', 
                  ax=axes[0][0])
plot_numeric_hist(nep_hhold, 
                  'nkid06', 
                  'Number of children 6 years old or under', 
                  ax=axes[0][1])
plot_numeric_hist(nep_hhold, 
                  'nfooditm_7', 
                  'Number of food items consumed in a week', 
                  ax=axes[0][2])
plot_numeric_hist(nep_hhold, 
                  'depratio3', 
                  'Dependent ratio', 
                  ax=axes[0][3])
plot_numeric_hist(nep_hhold, 
                  'nadultf', 
                  'Number of adult Females', 
                  ax=axes[1][0])
plot_numeric_hist(nep_hhold, 
                  'nelderly', 
                  'Number of elders age 65+', 
                  ax=axes[1][1])
plot_numeric_hist(nep_hhold, 
                  'nemp_fem', 
                  'Number of female household members employed', 
                  ax=axes[1][2])
plot_numeric_hist(nep_hhold, 
                  'nemp20', 
                  'Number of household members employed more than 20 hrs', 
                  ax=axes[1][3])

plt.show()

pd.DataFrame.from_dict(nep_hhold_q, orient='index')[100:150]

# Filter weekly consumption and group by poor/non-poor
consumption_columns = [x for x in nep_hhold.columns if x.startswith('sh_') and x.endswith('_7')]
consumption = (nep_hhold.groupby('poor')[consumption_columns].sum().T)

consumption.columns = ['Non_poor', 'Poor']
consumption['total'] = consumption.sum(axis=1)
consumption['percent'] = consumption.total / nep_hhold.shape[0]

# Match up the consumable names for readability
get_consumable_name = lambda x: nep_hhold_q[x.split('__')[0]]
consumption.index = consumption.index.map(get_consumable_name)


consumption['difference'] = (consumption.Non_poor - consumption.Poor) / consumption.total
display(consumption.sort_values('difference', ascending=False))

# Plot  share of weekly consumption as % difference in poor and non poor household
(consumption.difference.sort_values(ascending=False).sort_values(ascending=True).plot.barh());



# Filter monthly consumption and group by poor/non-poor
consumption_columns = [x for x in nep_hhold.columns if x.startswith('sh_') and x.endswith('_30')]
consumption = (nep_hhold.groupby('poor')[consumption_columns].mean().T)
consumption.columns = ['Non_poor', 'Poor']
consumption['total'] = consumption.sum(axis=1)
consumption['percent'] = consumption.total / nep_hhold.shape[0]
consumption['difference'] = (consumption.Non_poor - consumption.Poor)
display(consumption.sort_values('difference', ascending=False))

# Match up the consumable names for readability
get_consumable_name = lambda x: nep_hhold_q[x.split('__')[0]]
consumption.index = consumption.index.map(get_consumable_name)

consumption['difference'] = (consumption.Non_poor - consumption.Poor) / consumption.total
display(consumption.sort_values('total', ascending=False))

# Plot  share of weekly consumption as % difference in poor and non poor household
dd= pd.DataFrame(consumption.sort_values('difference', ascending=False))


(consumption.difference.sort_values(ascending=True).sort_values(ascending=True).plot.barh());
(consumption.Non_poor.sort_values(ascending=True).sort_values(ascending=True).plot.bar())




# Train and Test Split, with 25% data in test
print(nep_hhold.shape)
nep_train, nep_test = train_test_split(nep_hhold, test_size=0.25,random_state=1443,stratify=nep_hhold.poor)


#save the test and train data to files

nep_train.to_pickle("nepal_poverty_train.pkl")
nep_test.to_pickle("nepal_poverty_test.pkl")
with open("nepal_indv_questions.json", 'w') as fp:
    json.dump(nep_indv_q, fp)
with open("nepal_pov_questions.json", 'w') as fp:
    json.dump(nep_hhold_q, fp)

print('----------')
 

    







