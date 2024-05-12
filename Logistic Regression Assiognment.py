# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:57:44 2023

@author: sksha
"""
import numpy as np
import pandas as pd
df = pd.read_csv("bank-full order.csv")
df.info
df.shape
pd.set_option("display.max.columns",20)
df

# EDA
import seaborn as sns
sns.set_style(style="darkgrid")
sns.pairplot(df)

# Create boxplots for all columns using a for loop
for column in df.columns:
    plt.figure(figsize=(8, 4))  # Adjust the figure size as needed
    sns.boxplot(x=df[column], orient="v")
    plt.xlabel(column)
    plt.ylabel("Values")
    plt.title(f'Boxplot for {column}')
    plt.show()


continuous_columns = ["age", "balance","duration","campaign","pdays","previous" ]  # Add your column names here
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for df.cloumns in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_bound) & (data_without_outliers[column]<= upper_bound)]
# Print the cleaned data without outliers
print(data_without_outliers)
 
#Standardisation
X_cont=df[df.columns[[0,5,9,11,12,13,14]]]
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X1 = SS.fit_transform(X_cont)
X1 = pd.DataFrame(X1)
X1.columns=list(X_cont)
X1



df_cat = df.iloc[:,[1,2,3,4,6,7,8,10,15]]
df_cat
"""df_cat = df.drop(df.columns[[0,5,9,11,12,13,14]],axis=1)"""

#Label Encoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for i in range(0,9):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])
    
df_cat.head()
X = pd.concat([X1,df_cat],axis=1)
X


Y1 = LE.fit_transform(df["y"])
Y = pd.DataFrame(Y1)
Y

#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75)
X_train
X_test

#fit the model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred_train =logreg.predict(X_train)
Y_pred_test =logreg.predict(X_test)


#metrics
from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(Y,Y_pred)
print(cm)
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3))

training_accuracy = []
test_accuracy = []

for i in range(1,101):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=i)
    logreg.fit(X_train,Y_train)
    Y_pred_train =logreg.predict(X_train)
    Y_pred_test =logreg.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
print("Average Training accuracy ",np.mean(training_accuracy).round(3))
print("Average Test accuracy ",np.mean(test_accuracy).round(3))

from sklearn.metrics import recall_score,precision_score,f1_score
print("Sensitivity score:",recall_score(Y,Y_pred).round(3))
print("Precision score:",precision_score(Y,Y_pred).round(3))
print("F1 score:",f1_score(Y,Y_pred).round(3))

logreg.predict_proba(X)[:,1]
df["Y_probabilities"] = logreg.predict_proba(X)[:,1]

#Function to change the cut off
def f1(X):
    if X<0.4:
        return 0
    elif X>=0.4:
        return 1

df["Y_prob"] = df["Y_probabilities"].apply(f1)
df.head()

# ROC Curve plotting and finding AUC value
from sklearn.metrics import roc_auc_score,roc_curve
fpr,tpr,dummy = roc_curve(Y,df["Y_probabilities"])

import matplotlib.pyplot as plt
plt.scatter(x=fpr,y=tpr)
plt.plot(fpr,tpr,color='red')
plt.plot([0,1],[0,1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()
auc = roc_auc_score(Y,df["Y_probabilities"])
print("Area under curve:",(auc*100).round(3))