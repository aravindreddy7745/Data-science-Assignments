# -*- coding: utf-8 -*-
"""

@author: sksha
"""
""" importing the data """
import numpy as np
import pandas as pd
df = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
df
df.shape
df.head()
df.tail()
df.info()
df.isnull().sum()

# Split the 'Quarter' column into two columns
df[['Quarter', 'Year']] = df['Quarter'].str.split('_', expand=True)
df
# Prepend 's' to the 'Year' column
df['Year'] = 's' + df['Year']

print(df)

# Mapping of quarters to months
quarter_to_month = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}

# Create a new 'Month' column by mapping the 'Quarter' column
df['Month'] = df['Quarter'].map(quarter_to_month)

print(df)

df['Year'] = '19' + df['Year'].astype(str)
df

df['Month_Year'] = df['Month'].astype(str) + '/' + df['Year'].astype(str)

print(df)

# Extract month, year, and add '19' to the year

df['Month'] = df['Month_Year'].str.extract(r'(\d+)/')
df['Year'] = '19' + df['Month_Year'].str.extract(r's(\d+)$')
df
# Create a new 'Date' column by combining 'Month' and 'Year'
df['Date'] = pd.to_datetime(df['Month'] + df['Year'], format='%m%Y')

print(df)

#==================================================================================================
### EDA
### Exploratry Data Analysis

df.Sales.hist()

""" Creating the Dummy Variables """

# Extracting quarters and years from the 'Quarter' column and creating dummy variables
for i in range(42):
    p = df.loc[i, 'Quarter']
    df.loc[i, 'quarter'] = p[0:2]

for i in range(42):
    p = df.loc[i, 'Quarter']
    df.loc[i, 'Year'] = p[3:]

# Creating dummy variables for quarters
quarter_dummies = pd.get_dummies(df['quarter'])

# Concatenating dummy variables to the original DataFrame
df_Dummy = pd.concat([df, quarter_dummies], axis=1)

# Adding a time index ('t'), its square, and the log of 'Sales'
df_Dummy["t"] = np.arange(1, 43)
df_Dummy["t_square"] = np.square(df_Dummy["t"])
df_Dummy["Log_Sales"] = np.log(df_Dummy["Sales"])
df

#================================================================================================
# Correlation matrix
correlation_matrix = df_Dummy.corr()

# Heatmap for Correlation Matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()
#================================================================================================
# Creating a line plot for 'Sales' over time
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df_Dummy['t'], df_Dummy['Sales'], marker='o', linestyle='-', color='b')
plt.title('Sales Over Time')
plt.xlabel('Time (t)')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
#================================================================================================
# Creating a lag plot for 'Sales'
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
plt.figure(figsize=(8, 8))
lag_plot(df_Dummy['Sales'])
plt.title('Lag Plot for Sales')
plt.show()
#=================================================================================================
# Creating a density plot for 'Sales'
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.kdeplot(df_Dummy['Sales'], fill=True, color='skyblue')
plt.title('Density Plot for Sales')
plt.xlabel('Sales')
plt.ylabel('Density')
plt.show()
#================================================================================================
# Creating an autocorrelation plot for 'Sales'
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sm.graphics.tsa.plot_acf(df_Dummy['Sales'], lags=20)
plt.title('Autocorrelation Plot for Sales')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
#================================================================================================
# Creating a boxplot for 'Sales'
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_Dummy['quarter'], y=df_Dummy['Sales'])
plt.title('Boxplot of Sales by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.show()
#===============================================================================================
# Splitting the data
train=df_Dummy.head(int(len(df_Dummy.quarter)*0.9))
test=df_Dummy.tail(len(df_Dummy.quarter)-len(train))

train

test
#===============================================================================================

#===============================================================================================
#Linear Model
import statsmodels.formula.api as smf
Linear_model=smf.ols("Sales~t",data=train).fit()
pred_linear=Linear_model.predict(test["t"])
rmse_linear=np.sqrt(np.mean(test["Sales"]-np.array(pred_linear))**2)
rmse_linear       # 613.0885797796357


#Exponential
exp_model=smf.ols("Log_Sales~t",data=train).fit()
pred_exp_model=exp_model.predict(test["t"])
rmse_exp_model=np.sqrt(np.mean((test["Sales"])-(np.array(pred_exp_model)))**2)
rmse_exp_model    # 4713.8306396253665

#Quadratic
Quad=smf.ols("Sales~t+t_square",data=train).fit()
pred_Quad=Quad.predict(test[["t","t_square"]])
rmse_Quad=np.sqrt(np.mean((test["Sales"])-(np.array(pred_Quad)))**2)
rmse_Quad         # 37.603993337869

#Additive seasonality
Add_sea=smf.ols("Sales~Q1+Q2+Q3+Q4",data=train).fit()
pred_add_sea=Add_sea.predict(test[["Q1","Q2","Q3","Q4"]])
rmse_add_sea=np.sqrt(np.mean(test["Sales"]-(np.array(pred_add_sea)))**2)
rmse_add_sea      # 1916.793673290675

#Additive Seasonality Quadratic
add_sea_quad=smf.ols("Sales~t+t_square+Q1+Q2+Q3+Q4",data=train).fit()
pred_add_sea_quad=add_sea_quad.predict(test[["t","t_square","Q1","Q2","Q3","Q4"]])
rmse_add_sea_quad=np.sqrt(np.mean(test["Sales"]-np.array((pred_add_sea_quad)))**2)
rmse_add_sea_quad    # 108.6866689748922

# Multiplicative Seasonality
mul_sea=smf.ols("Log_Sales~Q1+Q2+Q3+Q4",data=train).fit()
pred_Mult_sea=mul_sea.predict(test[["Log_Sales","Q1","Q2","Q3","Q4"]])
rmse_Mult_sea=np.sqrt(np.mean(test["Sales"]-(np.array(np.exp(pred_Mult_sea))))**2)
rmse_Mult_sea        # 2009.5556325551393

#Multiplicative Additive Seasonality
mul_add_sea=smf.ols("Log_Sales~t+Q1+Q2+Q3+Q4",data=train).fit()
pred_mul_add_sea=mul_add_sea.predict(test)
rmse_mul_add_sea=np.sqrt(np.mean(test["Sales"]-(np.array(pred_mul_add_sea)))**2)
rmse_mul_add_sea     # 4713.803270264603

# The quadratic model (rmse_Quad) performed the best among the tested regression models, achieving the lowest root mean square error (RMSE) of approximately 37.60.

""" setting our Month column as index """
df.set_index('Date',inplace = True)
df.head()

""" observing how our initial data is plotted over graph to check seasonality( to be precise) """

df.plot()

""" as we can observe the graph overall the mean and standard deviation gets increasing and there is seasonality here 
in the depicted graph so there is no statinarity over here """

""" importing the necessary package for dickey fuller test  """

from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    if result[1] <= 0.05:
       print("Strong evidence against the null hypothesis,reject the null hypothesis.Data has no unit root and is stationary")
    else:
       print("Weak evidence against null hypothesis,time series has unit root , indicating it is non -statinary")


adf_test(df['Sales'])



#===================================================================================================
# First Differencing technique

df['Sales First Difference']=df['Sales']-df['Sales'].shift(1)
df.head()

adf_test(df['Sales First Difference'].dropna())    # data is still not stationary.


# Second Differencing technique

df['Sales Second Difference'] = df['Sales First Difference']-df['Sales First Difference'].shift(1)

adf_test(df['Sales Second Difference'].dropna())   # data is  stationary.

#=================================================================================================
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# First Differencing ACF and PACF
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['Sales First Difference'].dropna(), lags=10, ax=ax1)
plot_pacf(df['Sales First Difference'].dropna(), lags=10, ax=ax2)
ax1.set_title('ACF - First Differencing')
ax2.set_title('PACF - First Differencing')
plt.show()

# Second Differencing ACF and PACF
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['Sales Second Difference'].dropna(), lags=10, ax=ax1)
plot_pacf(df['Sales Second Difference'].dropna(), lags=10, ax=ax2)
ax1.set_title('ACF - Second Differencing')
ax2.set_title('PACF - Second Differencing')
plt.show()

#===================================================================================================
# ARIMA Model
from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(train['Sales'], order=(3, 2, 3))
arima_fit = arima_model.fit()

# Make predictions on the test set
arima_predictions = arima_fit.forecast(steps=len(test))

# Calculate RMSE
from sklearn.metrics import mean_squared_error
rmse_arima = np.sqrt(mean_squared_error(test['Sales'], arima_predictions))
print('RMSE (ARIMA):', rmse_arima)  # RMSE (ARIMA): 142.163236822853

# model summary
arima_fit.summary()

#==================================================================================================
# SARIMA Model
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima_model = SARIMAX(train['Sales'], order=(3, 2, 3), seasonal_order=(0, 0, 0, 12))
sarima_fit = sarima_model.fit()

# Make predictions on the test set
sarima_predictions = sarima_fit.forecast(steps=len(test))
sarima_predictions

# Calculate RMSE
from sklearn.metrics import mean_squared_error
rmse_sarima = np.sqrt(mean_squared_error(test['Sales'], sarima_predictions))
print('RMSE (SARIMA):', rmse_sarima)   #RMSE (SARIMA): 142.163236822853

# model summary
sarima_fit.summary()

#===================================================================================================
#Compare the results
data = {"MODEL":pd.Series(["rmse_linear","rmse_exp_model","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_mul_add_sea","rmse_ARIMA","rmse_SARIMAX"]),
        "RMSE_Values":pd.Series([rmse_linear,rmse_exp_model,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_mul_add_sea,rmse_arima,rmse_sarima])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

# Both the ARIMA and SARIMA models with the same order parameters (ARIMA(3, 2, 3) and SARIMA(3, 2, 3)) resulted in identical RMSE values of approximately 142.16 on the test set.
# The quadratic model (rmse_Quad) performed the best with the lowest RMSE value of approximately 37.60, indicating a good fit to the data.
# The ARIMA and SARIMAX models demonstrated similar performance with RMSE values of approximately 142.16
#===========================================================================================================================