# -*- coding: utf-8 -*-
"""

@author: sksha
"""

#load the data
import pandas as pd
df = pd.read_csv("Company_Data.csv")
df.shape #(400, 11)
df.info()

#==============================================================================
# Convert 'Sales' into a categorical variable (e.g., High Sales and Low Sales)
threshold = df['Sales'].quantile(0.75)  # You can adjust the threshold as needed
df['Sales'] = df['Sales'].apply(lambda x: 'High Sales' if x >= threshold else 'Low Sales')

#==============================================================================
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
data = df[df.columns[[1,2,3,4,5,7,8]]]
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#We can see there are some outliers present 

#Removing the outliers
df1 = df[df.columns[[1,2,3,4,5,7,8]]]
df1
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
import numpy as np
z_scores = np.abs(stats.zscore(df1))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(397, 11)

# Now, df contains the data with outliers removed

#===============================================================================
#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()
#======================================================================
#LabelEncoding for categorical Variables
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["ShelveLoc"] = LE.fit_transform(df["ShelveLoc"])
df["Urban"] = LE.fit_transform(df["Urban"])
df["US"] = LE.fit_transform(df["US"])

#=============================================================================
# Split the data into features (X) and the target variable (y)
X = df.drop("Sales", axis=1)
X
Y = df["Sales"]
Y
#==============================================================================
#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state=123)

#===============================================================================
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy')
DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train)
Y_pred_train
Y_pred_test = DT.predict(X_test)
Y_pred_test

#Metrices
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))  #ac1 = 1.0
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Testing Accuracy Score:",ac2.round(3))  #ac2 = 0.8

#Only runs in Google Colab
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(DT,filled= True,rounded=True,special_characters=True)
graph = graphviz.Source(dot_data) 

print("Number of nodes",DT.tree_.node_count)    #Number of nodes 65
print("Level of depth",DT.tree_.max_depth)      #Level of depth 9

#Validation Set Approach
training_accuracy = []
test_accuracy = []
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state=i)
    DT = DecisionTreeClassifier(criterion='gini',max_depth=9)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test = DT.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

import numpy as np
print("Average Training Accuracy:",np.mean(training_accuracy).round(3)) #Average Training Accuracy: 0.994
print("Average Test Accuracy:",np.mean(test_accuracy).round(3))         #Average Test Accuracy: 0.781

#==============================================================================
#Grid Search CV
# Import necessary libraries
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Create a Decision Tree classifier
DT = DecisionTreeClassifier(random_state=123)

# Create GridSearchCV object
grid_search = GridSearchCV(DT, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)   #Best Parameters: {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5}

# Print the best accuracy score found by GridSearchCV
print("Best Accuracy Score:", grid_search.best_score_)  #Best Accuracy Score: 0.7705435417884278

# Get the best model from the GridSearchCV
best_DT = grid_search.best_estimator_

# Predict using the best model
Y_pred_train = best_DT.predict(X_train)
Y_pred_test = best_DT.predict(X_test)

# Metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy Score:", ac1.round(3))   #Training Accuracy Score: 0.926
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy Score:", ac2.round(3))       #Test Accuracy Score: 0.83

#==============================================================================
#Parallel Ensemble Methods
#Bagging
from sklearn.ensemble import BaggingClassifier
Bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=7),
                        n_estimators=100,
                        max_samples=0.6,
                        max_features=0.7,
                        random_state=123)    
Bag.fit(X_train,Y_train)
Y_pred_train = Bag.predict(X_train)
Y_pred_test = Bag.predict(X_test)

#Metrices
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))   #Training Accuracy Score: 0.973
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3))   #Test Accuracy Score: 0.84

#======================================================================================
#Ada Boost (Adaptive Boosting Techniques)
from sklearn.ensemble import AdaBoostClassifier
ABC = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                         n_estimators=100,
                         learning_rate=0.1)
ABC.fit(X_train,Y_train)
Y_pred_train = ABC.predict(X_train)
Y_pred_test = ABC.predict(X_test)

#Metrices
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))   #Training Accuracy Score:1.0 
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3))   #Test Accuracy Score: 0.82

#=======================================================================================
#Visualisation
# Selecting a subset of features for pairplot
selected_features = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age']

# Pairplot for Selected Features:
sns.pairplot(df[selected_features])
plt.show()

## Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# The Decision Tree model achieved perfect accuracy (1.0) on the training set, indicating it perfectly fit the training data.
# there is a drop in accuracy on the test set (0.8), suggesting potential overfitting
'''The Bagging classifier, using Decision Trees with a maximum depth of 7, achieved high training accuracy (0.993) and slightly improved test accuracy (0.83).
Bagging helps reduce overfitting, as evidenced by the improved test accuracy compared to the standalone Decision Tree.'''

'''The AdaBoost classifier, using Decision Trees as base estimators, achieved perfect training accuracy (1.0) but lower test accuracy (0.77).
AdaBoost tends to be sensitive to noisy data, and the drop in test accuracy might indicate some overfitting or sensitivity to outliers.'''





#====================================================================================



