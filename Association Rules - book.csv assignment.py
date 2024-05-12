# -*- coding: utf-8 -*-
"""

@author: sksha
"""

import warnings
warnings.filterwarnings("ignore")
# Load the dataset
import pandas as pd
df = pd.read_csv("book.csv",encoding='latin-1')

#=================================================================================
# Exploratory Data Analysis (EDA)
# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display summary statistics of numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

#===================================================================================
# Visualize the distribution of numerical variables
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 8))
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.subplot(2, 2, 2)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()
#======================================================================================
# Visualize the correlation matrix for numerical variables
# This helps to identify relationships between variables
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
#====================================================================================
# Data Exploration
df.iloc[:,:].sum()
df.info()

for i in df.columns:
    print(i)
    print(df[i].value_counts())
    print()
    
#==================================================================================
"pip install wordcloud"
from wordcloud import WordCloud
plt.rcParams['figure.figsize'] = (10, 8)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(df.sum()))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()

#===================================================================================
# 1. Association rules with 10% Support and 30% confidence
from mlxtend.frequent_patterns import apriori,association_rules
movies_10_30 = apriori(df, min_support=0.1, use_colnames=True)
rules_10_30 = association_rules(movies_10_30, metric="confidence", min_threshold=0.3)

# Display the association rules
print("Association Rules with 10% Support and 30% Confidence:")
print(rules_10_30)

# Visualization of obtained rule
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rules_10_30['support'], y=rules_10_30['confidence'])
plt.title('Association Rules (10% Support, 30% Confidence)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()

movies1 = apriori(df, min_support=0.1, use_colnames=True)
movies1

rules = association_rules(movies1, metric="confidence", min_threshold=0.3)
rules

rules.sort_values('lift',ascending=False)
lift=rules[rules.lift>1]
lift

matrix = lift.pivot('antecedents', 'consequents', 'lift')
plt.figure(figsize=(12, 6), dpi=90)  # Adjusted the figsize
sns.heatmap(matrix, annot=True)
plt.title('HeatMap - For Lift Matrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])

# Apriori algorithm is used to mine association rules with a minimum support of 10% and a minimum confidence of 30%.
# Association rules are displayed along with support, confidence, and lift values.
# A scatter plot visualizes the support-confidence relationship.
#====================================================================================
# 2. Association rules with 15% Support and 50% confidence
movies2 = apriori(df, min_support=0.15, use_colnames=True)
movies2

rules = association_rules(movies2, metric="confidence", min_threshold=0.5)
rules

rules.sort_values('lift',ascending=False)
lift=rules[rules.lift>1]
lift

matrix = lift.pivot('antecedents','consequents','lift')
plt.figure(figsize=(20,6),dpi=60)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)

# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])

# Another set of association rules is mined with increased support (15%) and confidence (50%).
# Heatmap and scatter plot visualizations are provided for the obtained rules.
#=================================================================================
#3. Association rules with 17% Support and 40% confidence
movies3 = apriori(df, min_support=0.17, use_colnames=True)
movies3

rules = association_rules(movies3, metric="confidence", min_threshold=0.4)
rules

rules.sort_values('lift',ascending=False)
lift=rules[rules.lift>1]
lift

matrix = lift.pivot('antecedents','consequents','lift')
plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)

# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])

# Association rules are mined with different support (17%) and confidence (40%) thresholds.
# Heatmap and scatter plot visualizations are presented.

# Association rule mining is performed with varying support and confidence thresholds to discover interesting patterns in the data.
# The lift values in the association rules indicate the strength of relationships between items.
# Visualizations such as scatter plots and heatmaps help in better understanding the patterns and relationships revealed by the association rules.
#======================================================================================
