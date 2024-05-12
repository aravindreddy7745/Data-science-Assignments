# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:34:36 2023

@author: sksha
"""

import numpy as np
import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.dtypes
df.head()
list(df)
df["Profit"]
df["R&D Spend"]
df["Administration"]
df["Marketing Spend"]
df["State"]
df["State"].value_counts()

#Scatter plot
import matplotlib.pyplot as plt
plt.scatter(x=df['R&D Spend'], y=df['Profit'])         #1 = 0.972900
plt.scatter(x=df['Administration'], y=df['Profit'])    #3 = 0.200717
plt.scatter(x=df['Marketing Spend'], y=df['Profit'])   #2 = 0.747766
plt.show()
df.corr()

#Data Visualization
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)
sns

df = df.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
df

#Model Building
import statsmodels.formula.api as smf
model = smf.ols('Profit~RDS+ADMS+MKTS',data=df).fit()
model.summary()
model.fittedvalues    #predicted values
model.resid           #error values
model.resid.hist()
model.resid.skew()

#finding coefficient of parameters
model.params

#finding t-values and p-values
model.tvalues , np.round(model.pvalues,5)
model.rsquared , model.rsquared_adj

import statsmodels.formula.api as smf
import statsmodels.api as sm
slr_a=smf.ols("Profit~ADMS",data=df).fit()
slr_a.tvalues , slr_a.pvalues  # ADMS has in-significant pvalue

mlr_am=smf.ols("Profit~ADMS+MKTS",data=df).fit()
mlr_am.tvalues , mlr_am.pvalues  # varaibles have significant pvalues

#Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables
rsq_r=smf.ols("RDS~ADMS+MKTS",data=df).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMS~RDS+MKTS",data=df).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKTS~RDS+ADMS",data=df).fit().rsquared
vif_m=1/(1-rsq_m)

# Putting the values in Dataframe 
formatd1={'Variables':['RDS','ADMS','MKTS'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(df)
Vif_df

#Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()

list(np.where(model.resid<-30000))

Y = df["Profit"]
#X = df[['RDS']]
#X = df[['RDS','ADMS']]
#X = df[['RDS','ADMS','MKTS']]
#X = df[['RDS','ADMS','MKTS']]
X =df[["RDS","MKTS"]]


# fit the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y) # Bo + B1x1
LR.intercept_ # Bo
LR.coef_   #B1
# calc y_pred
Y_pred = LR.predict(X)
Y_pred
Y

# plt the scatter plot with y_pred
import matplotlib.pyplot as plt
plt.scatter(x = df['RDS'], y= df['Profit'])
plt.scatter(x= df['RDS'], y=Y_pred,color='red')
plt.plot(df[['RDS']], Y_pred,color='black')
plt.show()

#=============================================
# metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean squared Error:", mse.round(3))
print("Root Mean squared Error:", np.sqrt(mse).round(3))
   
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
print("R square:", r2.round(3))

