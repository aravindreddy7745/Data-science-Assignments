# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:15:40 2023

@author: sksha
"""

#Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

#1) Delivery_time -> Predict delivery time using sorting time 

#import
import pandas as pd
df = pd.read_csv("delivery_time.csv")
df
df.shape
df.info()

#EDA
#-->histogram
df["Delivery Time"].hist()
df["Delivery Time"].skew() #positively skewed
df["Delivery Time"].kurt() #slightly peaked

df["Sorting Time"].hist()
df["Sorting Time"].skew() #positively skewed
df["Sorting Time"].kurt() #platy kutosis

#-->scatter plot
import matplotlib.pyplot as plt
plt.scatter(x = df["Sorting Time"], y = df["Delivery Time"])
plt.show()
df[["Sorting Time","Delivery Time"]].corr() # x,y variables have positive correlation

#-->Box plot
#Sorting Time
df.boxplot(column = "Sorting Time",vert = False)
#-->therefore there were no outliers in out data

#-->Bar plot
df["Sorting Time"].value_counts()
t1 = df["Sorting Time"].value_counts()
t1.plot(kind="bar")

# X and Y variabls
X = df[["Sorting Time"]]
X 

#Transformations on X variable

import numpy as np
X["Sorting Time"] = X["Sorting Time"]**2
X["Sorting Time"] = X["Sorting Time"]**3

X["Sorting Time"] = np.sqrt(X["Sorting Time"])
X["Sorting Time"] = np.log(X["Sorting Time"])
X["Sorting Time"] = 1 / np.sqrt(X["Sorting Time"])


Y = df["Delivery Time"]
Y

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=5)
X_train
X_test
Y_train
Y_test

# Constructing Model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train) #b0+b1x1
LR.intercept_ #b0
LR.coef_ #b1
#-->prediction values
Y_pred_train = LR.predict(X_train)
Y_pred_train
Y_pred_test = LR.predict(X_test)
Y_pred_test

#metrics
import numpy as np
from sklearn.metrics import mean_squared_error
error = mean_squared_error(Y_train,Y_pred_train)
error
print("mean squared error :",error.round(3))
print("root squared error :",np.sqrt(error.round(3)))
#R^2 error
from sklearn.metrics import r2_score
r2 = r2_score(Y_train,Y_pred_train)
print("R square :",r2.round(3))

#X["Sorting Time"] = X["Sorting Time"]**2            |  #X["Sorting Time"] = X["Sorting Time"]**3
#mean squared error : 6.914                          |  #mean squared error : 7.691
#root squared error : 2.629448611401257              |  #root squared error : 2.773265223522626
#R square : 0.742                                    |  #R square : 0.713


#X["Sorting Time"] = np.sqrt(X["Sorting Time"])      |  #X["Sorting Time"] = np.log(X["Sorting Time"])
#mean squared error : 6.89                           |  #mean squared error : 6.703
#root squared error : 2.6248809496813372             |  #root squared error : 2.5890152568109754
#R square : 0.743                                    |  #R square : 0.75



#X["Sorting Time"] = 1 / np.sqrt(X["Sorting Time"])  |
#mean squared error : 7.863                          |
#root squared error : 2.8041041350135343             |
#R square : 0.707                                    |

#--> therefore in "square root Sorting Time" the root square error is low and R square is high .

#=====================================================================================================

# 2) Salary_hike -> Build a prediction model for Salary_hike

#import
import pandas as pd
df = pd.read_csv("Salary_Data.csv")
df
df.shape
df.info()

#EDA
#-->histogram
df["Salary"].hist()
df["Salary"].skew() #positively skewed
df["Salary"].kurt() #platy kurtosis

df["YearsExperience"].hist()
df["YearsExperience"].skew() #positively skewed
df["YearsExperience"].kurt() #platy kutosis

#-->scatter plot
import matplotlib.pyplot as plt
plt.scatter(x = df["YearsExperience"], y = df["Salary"])
plt.show()
df[["YearsExperience","Salary"]].corr() # x,y variables have positive correlation

#-->Box plot
df.boxplot(column = "YearsExperience",vert = False)
#-->therefore there were no outliers in out data

#-->Bar plot
df["YearsExperience"].value_counts()
t1 = df["YearsExperience"].value_counts()
t1.plot(kind="bar")

# X and Y variabls
X = df[["YearsExperience"]]

#Transformations on X variable
import numpy as np

X["YearsExperience"] = X["YearsExperience"]**2
X["YearsExperience"] = X["YearsExperience"]**3
X["SquareRootYearsExperience"] = np.sqrt(X["YearsExperience"])
X["LogYearsExperience"] = np.log(X["YearsExperience"])
X["InverseSquareRootYearsExperience"] = 1 / np.sqrt(X["YearsExperience"])

Y = df["Salary"]
Y

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=5)
X_train
X_test
Y_train
Y_test

# Constructing Model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train) #b0+b1x1
LR.intercept_ #b0
LR.coef_ #b1
#-->prediction values
Y_pred_train = LR.predict(X_train)
Y_pred_train
Y_pred_test = LR.predict(X_test)
Y_pred_test

#metrics
import numpy as np
from sklearn.metrics import mean_squared_error
error = mean_squared_error(Y_train,Y_pred_train)
error
print("mean squared error :",error.round(3))
print("root squared error :",np.sqrt(error.round(3)))
#R^2 error
from sklearn.metrics import r2_score
r2 = r2_score(Y_train,Y_pred_train)
print("R square :",r2.round(3))

#X["YearsExperience"] = X["YearsExperience"]**2      |  #X["YearsExperience"] = X["YearsExperience"]**3
#mean squared error : 56741956.63                    |  #mean squared error : 112073928.171
#root squared error : 7532.72571052471               |  #root squared error : 10586.49744585054
#R square : 0.916                                    |  #R square : 0.835


#X["YearsExperience"] = np.sqrt(X["YearsExperience"])|  #X["YearsExperience"] = np.log(X["YearsExperience"])
#mean squared error : 29179149.08                    |  #mean squared error : 28949608.93
#root squared error : 5401.772771970328              |  #root squared error : 5380.484080080528
#R square : 0.957                                    |  #R square : 0.957



#X["YearsExperience"] = 1 / np.sqrt(X["YearsExperience"])        |
#mean squared error : 5360.576772325903                          |
#root squared error : 2.8041041350135343                         |
#R square : 0.958                                                |

#--> therefore in "inverse square root yearsExperience" the root square error is low and R square is high .
