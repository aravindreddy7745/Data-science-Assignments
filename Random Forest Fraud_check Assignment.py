# -*- coding: utf-8 -*-
"""

"""
#load the data
import pandas as pd
import numpy as np
df = pd.read_csv("Fraud_check.csv")
df
df.info()
df.shape   #(600,6)
df.hist()
#=================================================================================
# Create the target variable
df["Taxable.Income"] = df["Taxable.Income"].apply(lambda x: "Risky" if x <= 30000 else "Good")
df["Taxable.Income"]
#=============================================================================
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
data = df.iloc[:,3:5]
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
    
#Removing the outliers
df1 = df.iloc[:,3:5]
df1
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
import numpy as np
z_scores = np.abs(stats.zscore(df1))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)
outlier_mask

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(600, 6)
# Outliers were removed from the dataset based on Z-scores.
# Now, df contains the data with outliers removed

#=====================================================================================
#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()

#=============================================================================
#Label Encoding 
# Encode the "Undergrad," "Marital.Status," and "Urban" categorical variables
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["Undergrad"] = LE.fit_transform(df["Undergrad"])
df["Marital.Status"] = LE.fit_transform(df["Marital.Status"])
df["Urban"] = LE.fit_transform(df["Urban"])

#=============================================================================
# Split the data into features (X) and the target variable (Y)
X = df.drop("Taxable.Income", axis=1)
X
Y = df["Taxable.Income"]
Y
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y1=LE.fit_transform(df["Taxable.Income"])
Y1
#==============================================================================
#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y1,train_size = 0.75,random_state=123)

#==============================================================================
#Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_depth=7,
                        n_estimators=100,
                        max_samples=0.6,
                        max_features=0.7,
                        random_state=123)    
RF.fit(X_train,Y_train)
Y_pred_train = RF.predict(X_train)
Y_pred_test = RF.predict(X_test)

#Metrices
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",(ac1*100).round(3))  #Training Accuracy Score: 82.444
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",(ac2*100).round(3))  #Test Accuracy Score: 79.333

# Confusion matrix and classification report
confusion = confusion_matrix(Y_test, Y_pred_test)
classification_rep = classification_report(Y_test, Y_pred_test)

print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)

#=================================================================================
#Grid Search CV
# Import necessary libraries
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 7, 9],
    'max_samples': [0.6, 0.7, 0.8],
    'max_features': [0.6, 0.7, 0.8]
    }

# Create a Random Forest classifier
RF = RandomForestClassifier(random_state=123)

# Create GridSearchCV object
grid_search = GridSearchCV(RF, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)      #Best Parameters: {'max_depth': 7, 'max_features': 0.8, 'max_samples': 0.7, 'n_estimators': 50}

# Print the best accuracy score found by GridSearchCV
print("Best Accuracy Score:", grid_search.best_score_)   #Best Accuracy Score: 0.7955555555555556

# Get the best model from the GridSearchCV
best_RF = grid_search.best_estimator_

# Predict using the best model
Y_pred_train = best_RF.predict(X_train)
Y_pred_test = best_RF.predict(X_test)

# Metrics
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy Score:", (ac1*100).round(3))  #Training Accuracy Score: 83.11
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy Score:", (ac2*100).round(3))      #Test Accuracy Score: 80.0

#when i applied Random Forest classifier , i got Training Accuracy Score: 82 ,Test Accuracy Score: 79%
# I gone through hyperparameter tuning process with GridSearchCV, the algorithm will systematically evaluate different combinations of these hyperparameter values to find the combination
#I got Best Parameters: {'max_depth': 7, 'max_features': 0.8, 'max_samples': 0.7, 'n_estimators': 50}
#Training Accuracy Score: 0.831
#Test Accuracy Score: 80.0  it is good model to be considered

#=====================================================================================
# Visualisation
# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Adjusting hyperparameters or exploring other algorithms may be necessary to enhance the model's performance.
# The model achieved a training accuracy of 82.44% and a test accuracy of 79.33%. The relatively close values suggest that the model is not overfitting.
# The confusion matrix shows that the model correctly identified 119 instances of "Good" (class 0) but struggled to identify any instances of "Risky" (class 1), resulting in a low recall for class 1.
#  The classification report indicates that the model's precision for the "Risky" class (1) is 0%, meaning it predicted all instances as "Good" (class 0).
# The confusion matrix heatmap visually emphasizes the model's difficulty in correctly identifying instances of the "Risky" class






























