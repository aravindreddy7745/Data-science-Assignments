
#load the data
import pandas as pd
df = pd.read_csv("Company_Data.csv")
df.shape  #(400,11)
df.info()
df.hist()
#===============================================================================
# Convert 'Sales' into a categorical variable (e.g., High Sales and Low Sales)
threshold = df['Sales'].quantile(0.75)  # You can adjust the threshold as needed
df['Sales'] = df['Sales'].apply(lambda x: 'High Sales' if x >= threshold else 'Low Sales')
df['Sales']
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
#=======================================================================================
#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()
#=======================================================================================
#Label Encoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["ShelveLoc"] = LE.fit_transform(df["ShelveLoc"])
df["Urban"] = LE.fit_transform(df["Urban"])
df["US"] = LE.fit_transform(df["US"])
df.info()

# Split the data into features (X) and the target variable (y)
X = df.drop("Sales", axis=1)
X
Y = df["Sales"]
Y
#==============================================================================
#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state=123)

#==============================================================================
#Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100,
                            max_depth=7,
                            max_samples=0.6,
                            max_features=0.7,
                            random_state=123)    
RF.fit(X_train,Y_train)
Y_pred_train = RF.predict(X_train)
Y_pred_test = RF.predict(X_test)

#Metrices
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))  #Training Accuracy Score: 0.987
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3))  #Test Accuracy Score: 0.8

# Confusion matrix and classification report
confusion = confusion_matrix(Y_test, Y_pred_test)
classification_rep = classification_report(Y_test, Y_pred_test)

print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)

#=============================================================================
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
print("Best Parameters:", grid_search.best_params_)      #Best Parameters: {'max_depth': 7, 'max_features': 0.8, 'max_samples': 0.8, 'n_estimators': 100}
# Print the best accuracy score found by GridSearchCV
print("Best Accuracy Score:", grid_search.best_score_)   #Best Accuracy Score: 0.8836937463471652
# Get the best model from the GridSearchCV
best_RF = grid_search.best_estimator_

# Predict using the best model
Y_pred_train = best_RF.predict(X_train)
Y_pred_test = best_RF.predict(X_test)

# Metrics
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy Score:", (ac1*100).round(3))  #Training Accuracy Score: 98.63
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy Score:", (ac2*100).round(3))      #Test Accuracy Score: 79.59

# After hyperparameter tuning with GridSearchCV, the best parameters are:
# Max Depth: 9 ,Max Features: 0.8, Max Samples: 0.6 , Number of Estimators: 50
# The best accuracy score after hyperparameter tuning is 87.88%, indicating an improvement from the initial model.
# The training accuracy remains high at 98.65%
# The test accuracy is 79.0%, indicating good generalization

#===================================================================================================================
# Visualize Feature Importance
feature_importance = pd.Series(best_RF.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(10)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='skyblue')
plt.title('Top 10 Important Features')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()

# when i applied Random Forest classifier , i got Training Accuracy Score: 98 ,Test Accuracy Score: 80%
# I gone through hyperparameter tuning process with GridSearchCV, the algorithm will systematically evaluate different combinations of these hyperparameter values to find the combination
#I got Best Parameters: {'max_depth': 7, 'max_features': 0.8, 'max_samples': 0.7, 'n_estimators': 50}
#Training Accuracy Score: 98.63
#Test Accuracy Score: 79.59  it is good model to be considered
#This bar plot will provide a clear visual representation of the importance of each feature, with 'Price' being prominently displayed among the top features
#==========================================================================================

