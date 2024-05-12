# -*- coding: utf-8 -*-
"""

@author: sksha
"""

# Load the dataset
import pandas as pd
df = pd.read_csv('wine.csv')
df.info()
df.shape  # (178, 14)   

#=================================================================================
# Exploratory Data Analysis
# Define the columns to check for outliers (excluding 'Type' column)
columns_to_check = df.columns[1:]
columns_to_check
# Function to remove outliers using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Remove outliers
data_no_outliers = remove_outliers_iqr(df, columns_to_check)
data_no_outliers  #[161 rows x 14 columns]
# Reset the index
data_no_outliers = data_no_outliers.reset_index(drop=True)

# Check the shape of the dataset after removing outliers
print("Shape of the dataset after removing outliers:", data_no_outliers.shape)  # (161, 14)

df = data_no_outliers
df
#==================================================================================
# Transformation
# Extract the features (excluding the 'Type' column)
df_cont = df.drop('Type', axis=1)
df_cont
# Standardize the data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(df_cont)
X = pd.DataFrame(SS_X)
X.columns = list(df_cont)

#==================================================================================
# Principal Component Analysis
# Perform PCA to get the first three principal components
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X)
principal_components = pd.DataFrame(principal_components)
principal_components    # [161 rows x 3 columns]
explained_variance = pca.explained_variance_ratio_
explained_variance

#===================================================================================
# Hierarchical clustering
# Printing the Hierarchical Linkage Matrix:
from scipy.cluster.hierarchy import linkage
linkage_matrix = linkage(principal_components, method='ward')
print("Hierarchical Linkage Matrix:")
print(linkage_matrix)

#=================================================================================
# Plotting a Dendrogram (Visual Representation):
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
linkage_matrix = linkage(principal_components, method='ward')

# Create a dendrogram
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()


""" in our code , we are using the hierarchial(aggloromative) clustering to group our data points.
however, the no of clusters we specified is not directly impact the dendogram as the dendogram shows the hierarchial clusterng of the data
and does not directly display the specified no of clusters.to visualize with the correct number of clusters ,we should adjust our linkage method to "single" accordingly """

#==================================================================================
# Create and fit an Agglomerative Clustering model
# Forming a group using clustering
from sklearn.cluster import AgglomerativeClustering
n_clusters = 3  # You can adjust the number of clusters as needed
agglomerative = AgglomerativeClustering(n_clusters=n_clusters,affinity='euclidean',linkage='complete')
agglomerative_labels = agglomerative.fit_predict(X)
Y_new = pd.DataFrame(agglomerative_labels)
Y_new[0].value_counts()

#==================================================================================
# Visualize the clustered data
# Visualize the first and second principal components
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=agglomerative_labels, palette='viridis', s=50)
plt.title('Agglomerative Clustering')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.legend(title='Cluster')
plt.show()

#==================================================================================
# Visualize the first and third principal components
sns.set(font_scale=1)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 2], hue=agglomerative_labels, palette='viridis', s=50)
plt.title('Agglomerative Clustering')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[2])
plt.legend(title='Cluster')
plt.show()

#===================================================================================
# Visualize the second and third principal components
sns.set(font_scale=1)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 1], y=X.iloc[:, 2], hue=agglomerative_labels, palette='viridis', s=50)
plt.title('Agglomerative Clustering')
plt.xlabel(X.columns[1])
plt.ylabel(X.columns[2])
plt.legend(title='Cluster')
plt.show()

#==================================================================================================================
# K-means clustering with the elbow method
from sklearn.cluster import KMeans
kresults = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(principal_components)
    kresults.append(kmeans.inertia_)

# Plot the elbow curve
import matplotlib.pyplot as plt
plt.plot(range(1, 11),kresults, marker='o')
plt.title('Elbow Curve for K-Means Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

"""according to the elbow method we can get clarity on upto  which k value should be choosen"""
"""  at a certain stage ,we will able to see minimal drop of inertia values from major drop of inertial value.those minimal drop 
inertial k-stages can be neglected or ignored"""

""" here in this case we have a sequence of minimal drop of inertia values from k=5 onwards so we can neglect them and we will choose
the k value as 3 in this case"""

""" we have obtained the same clusters as the original data so this is the optimal solution  """


# From the elbow curve, determine the optimal number of clusters
optimal_clusters = 3  # Determine based on the elbow curve

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
kmeans_clusters = kmeans.fit_predict(principal_components)
kmeans_clusters

# Compare the obtained clusters to the original 'Type' column
original_clusters = df['Type']
original_clusters

# Print the results
print("Hierarchical Clustering Labels:")
print(linkage(principal_components, method='ward')[:,2])
print("Agglomerative Clustering Labels:")
print(agglomerative_labels)
print("\nK-Means Clustering Labels:")
print(kmeans_clusters)
print("\nOriginal 'Type' Labels:")
print(original_clusters)

# the Adjusted Rand Index to measure the similarity between the K-Means clusters and the original 'Type' labels. A higher ARI indicates a better clustering performance
# Adjusted Rand Index (ARI)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(original_clusters, kmeans_clusters)
print(f'Adjusted Rand Index: {ari}')   # Adjusted Rand Index: 0.9471506150594584
# This will provide you with a quantitative measure of the similarity between the obtained K-Means clusters and the original 'Type' labels. The closer the ARI is to 1, the better the clustering results align with the original classes.
#=========================================================================================