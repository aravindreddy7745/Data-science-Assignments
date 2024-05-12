# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:24:53 2023

@author: sksha
"""

#import the data
import numpy as np
import pandas as pd
df1_train = pd.read_csv("SalaryData_Train.csv")
df1_train.shape    #(30161, 14)
df1_train
df2_test = pd.read_csv("SalaryData_Test.csv")
df2_test
df2_test.shape    #(15060, 14)
df1_train.info()   
df2_test.info()

#========================================================================================================
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
data = df1_train[df1_train.columns[[0,3,9,10,11]]]
data
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df1_train[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
df1_cont = df1_train[df1_train.columns[[0,3,9,10,11]]]
df1_cont.shape  #(30161, 5)
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
import numpy as np
z_scores = np.abs(stats.zscore(df1_cont))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df1_train = df1_train[~outlier_mask]
df1_train.shape  #(26752, 14)
df1_train.info()
#=========================================================================================================
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
data = df2_test[df2_test.columns[[0,3,9,10,11]]]
data
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df2_test[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
df2_cont = df2_test[df2_test.columns[[0,3,9,10,11]]]
df2_cont.shape  #(15060, 5)
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
import numpy as np
z_scores = np.abs(stats.zscore(df2_cont))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df2_test = df2_test[~outlier_mask]
df2_test.shape  #(13903, 14)
#=============================================================================================================
# Assuming your target variable column is named 'target' in both DataFrames
X_train = df1_train.drop(columns=['Salary'])  # Features for training data
Y_train = df1_train['Salary']               # Target variable for training data

X_test = df2_test.drop(columns=['Salary'])    # Features for testing data
Y_test = df2_test['Salary']                   # Target variable for testing data

#====================================================================================================
# Apply label encoding to categorical columns
categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for column in categorical_columns:
    X_train[column] = LE.fit_transform(X_train[column])
    X_test[column] = LE.transform(X_test[column]) 
    
#======================================================================================================

#=====================================================================================
#model fitting using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,Y_train)
Y_pred_train = mnb.predict(X_train)
Y_pred_train 
Y_pred_test = mnb.predict(X_test)
Y_pred_test

#metrics
from sklearn.metrics import accuracy_score
AC1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score : ",AC1.round(3))
AC2 = accuracy_score(Y_test,Y_pred_test)
print("Testing Accuracy score : ",AC2.round(3))

#Training Accuracy score :  0.786
#Testing Accuracy score :  0.787     

#=======================================================================================






