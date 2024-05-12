#load the data
import pandas as pd
df = pd.read_csv("Fraud_check.csv")
df
df.info()
df.shape  #(600, 6)

#========================================================================================
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

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(600, 6)
# Now, df contains the data with outliers removed

#=========================================================================
#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()

#=============================================================================
#Label Encoding # Encode the "Undergrad," "Marital.Status," and "Urban" categorical variables
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["Undergrad"] = LE.fit_transform(df["Undergrad"])
df["Marital.Status"] = LE.fit_transform(df["Marital.Status"])
df["Urban"] = LE.fit_transform(df["Urban"])


# Split the data into features (X) and the target variable (y)
X = df.drop("Taxable.Income", axis=1)
X
Y = df["Taxable.Income"]
Y

#==============================================================================
#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state=123)

#===============================================================================
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini',max_depth=9)
DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train)
Y_pred_train
Y_pred_test = DT.predict(X_test)
Y_pred_test

#Metrices
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))  #ac1 = 0.893
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Training Accuracy Score:",ac2.round(3))  #ac2 = 0.74

#Graphviz
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(DT,filled= True,rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)

print("Number of nodes",DT.tree_.node_count)    #Nodes = 125
print("Level of depth",DT.tree_.max_depth)      #Level of depth = 9

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
print("Average Training Accuracy:",np.mean(training_accuracy).round(3)) #Average Training Accuracy: 0.881
print("Average Test Accuracy:",np.mean(test_accuracy).round(3))         #Average Test Accuracy: 0.712

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
print("Training Accuracy Score:",ac1.round(3))   #Training Accuracy Score: 0.8
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3))   #Test Accuracy Score: 0.793

#==============================================================================
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
print("Test Accuracy Score:",ac2.round(3))   #Test Accuracy Score: 0.667

#===============================================================================
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Visualizing Boxplots
data = df.iloc[:, 3:5]
for column in data:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Box Plot of {column}")
    plt.show()

# Confusion Matrix for Decision Tree
conf_matrix_dt = confusion_matrix(Y_test, Y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['Risky', 'Good'], yticklabels=['Risky', 'Good'])
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature Importance Plot for Decision Tree
feature_importance = DT.feature_importances_
features = X.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=features, orient='h', palette='viridis')
plt.title('Feature Importance - Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
# The model is giving high importance to the "City.Population" variable, indicating that the population size of the city where an individual resides is crucial in predicting their taxable income status.

# Confusion Matrix for Bagging
conf_matrix_bag = confusion_matrix(Y_test, Y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues', xticklabels=['Risky', 'Good'], yticklabels=['Risky', 'Good'])
plt.title('Confusion Matrix - Bagging')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Confusion Matrix for AdaBoost
conf_matrix_ada = confusion_matrix(Y_test, Y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_ada, annot=True, fmt='d', cmap='Blues', xticklabels=['Risky', 'Good'], yticklabels=['Risky', 'Good'])
plt.title('Confusion Matrix - AdaBoost')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# The decision tree model shows a good fit to the training data but might be slightly overfitting.
# Bagging has increased the model's stability and slightly improved test accuracy compared to the standalone decision tree.
# AdaBoost achieves perfect training accuracy, but it seems to struggle with generalization to the test set, indicating potential overfitting.
 
