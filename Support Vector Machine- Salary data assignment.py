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
#EDA----->EXPLORATORY DATA ANALYSIS on Train Data
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
#EDA----->EXPLORATORY DATA ANALYSIS on the Test Data
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
# Standardize the data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.transform(X_test)
#This code ensures that both training and testing data are label-encoded and standardized correctly before fitting the SVM model.

#===================================================================================================
# Create and train your SVM model
#Linear Function
from sklearn.svm import SVC         #Linear
svc = SVC(C=1.0, kernel='linear')
svc.fit(X_train, Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training accuracy score:", ac1.round(3))    #Train accuracy score: 0.806
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Testing accuracy score:", ac2.round(3))     #Test accuracy score: 0.805

"""when we have two distinct datasets,one for training and one for testing,the validation approach involving cross validation
 may not be directly applicable.The purpose of cross validation is to utilize the training data for model validationa and as 
 a tuning parameter.so we will split the train data into train set and validation set, fit our model on the train data and evaluate the model performance using the validation 
 data ,once we have selected the best performing model we will evaluate on the independent test data set to estimate the models performance on unseen data. """

# Assuming we have X_train, Y_train, X_test, Y_test
# Step 1: Split the training data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.75, random_state=42)

# Step 2: Fit our model on the training set and evaluating  on the validation set
from sklearn.svm import SVC
svc = SVC(C=1.0, kernel='linear') # linear function
svc.fit(X_train, Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_val = svc.predict(X_val)

# Step 3: Evaluating  the performance on the validation set
from sklearn.metrics import accuracy_score
ac_val = accuracy_score(Y_val, Y_pred_val)
print("Validation accuracy score:", ac_val.round(3))   #Validation accuracy score: 0.805

# Step 4: Once we have selected the best model,  we evaluate it on the independent test dataset
Y_pred_test = svc.predict(X_test)
ac_test = accuracy_score(Y_test, Y_pred_test)
print("Test accuracy score:", ac_test.round(3))      #Test accuracy score: 0.806
#Validation accuracy score: 0.805
#Test accuracy score: 0.806

#===================================================================================================
##Polynomial function
#SVM
from sklearn.svm import SVC
svc =SVC(degree=3,kernel='poly')  #Polynomial function
svc.fit(X_train,Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("training accuracy score:",(ac1*100).round(3))    #training accuracy score: 84.733
ac2 = accuracy_score(Y_test,Y_pred_test)
print("testing accuracy score:",(ac2*100).round(3))     #testing accuracy score: 84.277

# Assuming we have X_train, Y_train, X_test, Y_test
# Step 1: Split the training data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.75, random_state=42)

# Step 2: Fit our model on the training set and evaluating  on the validation set
from sklearn.svm import SVC
svc = SVC(degree=3, kernel='poly') # linear function
svc.fit(X_train, Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_val = svc.predict(X_val)

# Step 3: Evaluating  the performance on the validation set
from sklearn.metrics import accuracy_score
ac_val = accuracy_score(Y_val, Y_pred_val)
print("Validation accuracy score:", ac_val.round(3))      #Validation accuracy score: 0.834

# Step 4: Once we have selected the best model,  we evaluate it on the independent test dataset
Y_pred_test = svc.predict(X_test)
ac_test = accuracy_score(Y_test, Y_pred_test)
print("Test accuracy score:", ac_test.round(3))     #Test accuracy score: 0.842

#========================================================================================================
#Radial Basis Function
#SVM
from sklearn.svm import SVC
svc =SVC(degree=3,kernel='rbf')  #Radial Basis Function
svc.fit(X_train,Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("training accuracy score:",(ac1*100).round(3))    #training accuracy score: 85.075
ac2 = accuracy_score(Y_test,Y_pred_test)
print("testing accuracy score:",(ac2*100).round(3))       #testing accuracy score: 84.787

# Assuming we have X_train, Y_train, X_test, Y_test
# Step 1: Split the training data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.75, random_state=42)

# Step 2: Fit our model on the training set and evaluating  on the validation set
from sklearn.svm import SVC
svc = SVC(degree=3, kernel='rbf') # Radial Basis Function
svc.fit(X_train, Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_val = svc.predict(X_val)

# Step 3: Evaluating  the performance on the validation set
from sklearn.metrics import accuracy_score
ac_val = accuracy_score(Y_val, Y_pred_val)
print("Validation accuracy score:", ac_val.round(3))

# Step 4: Once we have selected the best model,  we evaluate it on the independent test dataset
Y_pred_test = svc.predict(X_test)
ac_test = accuracy_score(Y_test, Y_pred_test)
print("Test accuracy score:", ac_test.round(3))
#Validation accuracy score: 0.856
#Test accuracy score: 0.846

#=====================================================================================
# Model evaluation
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
Y_pred = svc.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# The SVM model achieved an accuracy of approximately 84.6% on the test data. This means that the model correctly predicted the salary category for about 84.6%
''' For the "<=50K" category (individuals with salaries less than or equal to $50K):
 Precision (accuracy of positive predictions): 86% , Recall (true positive rate): 95% , F1-score (harmonic mean of precision and recall): 91%'''

'''For the ">50K" category (individuals with salaries greater than $50K):
Precision: 74%
Recall: 49%
F1-score: 59%'''

#The model performs better in predicting the "<=50K" category, with a higher precision, recall, and F1-score. This indicates that the model is more accurate in identifying individuals with salaries less than or equal to $50K.

#===========================================================================================================












































































































