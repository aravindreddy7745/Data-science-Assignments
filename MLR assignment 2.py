# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:02:42 2023

@author: sksha
"""

#importing the data
import numpy as np
import pandas as pd
df = pd.read_csv("ToyotaCorolla.csv",encoding = 'latin')
df.head()

df = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
df
df.dtypes

#EDA
df.columns
df.info()
 
df[df.duplicated()]
df = df.drop_duplicates().reset_index(drop=True)
df.describe()
df

#correlation analysis
df.corr()

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)
sns

#Model Building
import statsmodels.formula.api as smf
model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()
model.summary()
model.fittedvalues    #predicted values
model.resid           #error values
model.resid.hist()
model.resid.skew()

#finding coefficient of parameters
model.params

#finding t-values and p-values
model.tvalues , np.round(model.pvalues,5)      #model Accuracy
model.rsquared , model.rsquared_adj         # (0.8625200256947001, 0.8617487495415147)

#SLR Model
slr_c=smf.ols('Price~cc',data=df).fit()
slr_c.tvalues , slr_c.pvalues

slr_d=smf.ols('Price~Doors',data=df).fit()
slr_d.tvalues , slr_d.pvalues 

#MLR Model
mlr_cd=smf.ols('Price~cc+Doors',data=df).fit()
mlr_cd.tvalues , mlr_cd.pvalues

'''rsq_age=smf.ols('Age~KM+HP+cc+Doors+Gears+QT+Weight',data=df).fit().rsquaredvif_age=1/(1-rsq_age)'''
#Residual Analysis(Q-Q plot)
import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot = sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()

list(np.where(model.resid>10))

list(np.where(model.resid>6000))


list(np.where(model.resid<6000))







 




