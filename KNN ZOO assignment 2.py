# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:11:59 2023

@author: sksha
"""

import numpy as np
import pandas as pd
df = pd.read_csv("glass (1).csv")
df.head()
df.shape

X = df.iloc[:,0:9]
X 
Y = df["Type"]
Y

#DATA PARTITION
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75)

#KNN
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor(n_neighbors=5,p=2)
KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

#METRICS ACCURACY SCORE
from sklearn.metrics import mean_squared_error
ac1 = mean_squared_error(Y_train,Y_pred_train)
print("KNN-Training error:",np.sqrt(ac1).round(3))
ac2 = mean_squared_error(Y_test,Y_pred_test)
print("KNN-Test error:",np.sqrt(ac2).round(3))

#VALIDTAION APPROACH FOR KNN
l1 = [] 
l2 = []
training_error=[]
test_error=[]
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=i)
    KNN = KNeighborsRegressor(n_neighbors=5,p=2,) #best k value = 
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    training_error.append(mean_squared_error(Y_train,Y_pred_train))
    test_error.append(mean_squared_error(Y_test,Y_pred_test))
print("Average Trianing error :",np.sqrt(training_error).round(3))  #
print("Average Test error :",np.sqrt(test_error).round(3))          #

#Average accuracies are getting stored 
l1.append(np.sqrt(training_error).round(3))
l2.append(np.sqrt(test_error).round(3))
print(l1)
print(l2)

#subtracting two list by converting into arrays
l1 
l2  

array1= np.array(l1)
array1
array2= np.array(l2)
array2
deviation = np.subtract(array1,array2)
deviation
list(deviation.round(3))



#Graph between accuracy score and k-value
import matplotlib.pyplot as plt
plt.scatter(range(5,17,2),l1)
plt.plot(range(5,17,2),l1,color='black')
plt.scatter(range(5,17,2),l2,color='red')
plt.plot(range(5,17,2),l2,color='black')
plt.show()