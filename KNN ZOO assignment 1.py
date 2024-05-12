# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:36:09 2023

@author: sksha
"""
import numpy as np
import pandas as pd
df = pd.read_csv("Zoo.csv")
df.head()
df.shape

X = df.iloc[:,1:16]
X

Y = df["type"]
Y

#DATA PARTITION
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75)

#KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5,p=2) #5,7,9,11,13,15
KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

#METRICS ACCURACY SCORE
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training accuracy:",ac1.round(3))
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test accuracy:",ac2.round(3))

#VALIDTAION APPROACH FOR KNN
l1 = []
l2 = []
training_accuracy=[]
test_accuracy=[]
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=i)
    KNN = KNeighborsClassifier(n_neighbors=11,p=2) #best k value = 11
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
print("Average Trianing accuracy :",np.mean(training_accuracy))  #0.7888215488215486
print("Average Test accuracy :",np.mean(test_accuracy))          #0.7680652680652679

#Average accuracies are getting stored 
l1.append(np.mean(training_accuracy))
l2.append(np.mean(test_accuracy))
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
