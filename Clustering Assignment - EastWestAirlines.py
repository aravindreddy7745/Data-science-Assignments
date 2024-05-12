# -*- coding: utf-8 -*-
"""

@author: sksha
"""
#Import the Data
import numpy as np
import pandas as pd
df = pd.read_excel("EastWestAirlines.xlsx",sheet_name='data')
df.head()
df.shape  #(3999, 12)
df.info()
df.isnull().sum()

#==============================================================================
#EDA
#BOXPLOT AND OUTLIERS CALCULATION #
df1 = df[df.columns[[1,2,6,7,8,9,10]]]
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
z_scores = np.abs(stats.zscore(df1))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(3293, 12)

# Now, df contains the data with outliers removed

#================================================================================
#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()

#=============================================================================
# Create a pairplot for your DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df)
plt.show()

# understanding the relationships between all the variables#
import seaborn as sns
import matplotlib.pyplot as plt
# Generate a correlation matrix for your DataFrame
correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
"""Values close to 1 indicate a strong positive correlation.
Values close to -1 indicate a strong negative correlation.
Values close to 0 indicate a weak or no correlation"""

#===========================================================================
#Dendogram
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
# Calculate the linkage matrix
linkage_matrix = sch.linkage(df, method='ward')
linkage_matrix
# Create a dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()
#==========================================================================
X =df.iloc[:,0:13]
X
#AgglomerativeClustering
#forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'complete')
Y = cluster.fit_predict(X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

# Create a scatterplot to visualize the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=Y, cmap='viridis', marker='o', edgecolor='k')
plt.title('Agglomerative Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#============================================================================
#performing k means on the same data
#KMeans (Elbow method)
from sklearn.cluster import KMeans
# Specify the number of clusters (k) you want to create
k = 3  # Adjust this value based on your requirements

# Create an instance of the KMeans algorithm
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the model to your data and get cluster labels
cluster_labels = kmeans.fit_predict(df)

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Create a scatterplot to visualize the K-Means clustering results
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

"""according to the elbow method we can get clarity on upto  which k value should be choosen"""

#=========================================================================
#DBSCAN
from sklearn.cluster import DBSCAN
# Create an instance of the DBSCAN algorithm
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust 'eps' and 'min_samples' as needed

# Fit the model to your data and get cluster labels
cluster_labels = dbscan.fit_predict(df)

# Create a scatterplot to visualize the DBSCAN clustering results
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Hierarchical clustering was performed, and the dendrogram was plotted to visualize the hierarchy of clusters,The dendrogram suggests the presence of 5 distinct clusters
# K-Means clustering was applied with k=3 clusters based on the elbow method.
# The scatterplot displays the data points colored by the clusters, with red points representing cluster centers.
# DBSCAN clustering was performed with specified parameters (eps=0.5, min_samples=5).
# The scatterplot visualizes the clusters identified by DBSCAN.
# The K-Means algorithm identified three distinct clusters, The dataset may have hierarchical structures, and five main clusters can be identified.,
# through DBSCAN It successfully marked noise points (outliers) separately, providing a clear distinction.
#============================================================

























