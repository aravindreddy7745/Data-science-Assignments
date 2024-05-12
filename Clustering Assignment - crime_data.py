import numpy as np
import pandas as pd
df = pd.read_csv("crime_data.csv")
df
df.info()
df.isnull().sum()
pd.set_option('display.max_columns', None)
df
df.shape #(50, 5)

#=================================================================================
# EDA #
#EDA
#BOXPLOT AND OUTLIERS CALCULATION #
df1 = df.iloc[:,1:5]
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
z_scores = np.abs(stats.zscore(df1))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(3630, 12)

# Now, df contains the data with outliers removed

#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
import seaborn as sns
df["Murder"].hist()
sns.distplot(df["Murder"])
df["Murder"].skew()
df["Murder"].kurt()
df["Murder"].describe()

df["Assault"].hist()
sns.distplot(df["Assault"])
df["Assault"].skew()
df["Assault"].kurt()
df["Assault"].describe()

df["UrbanPop"].hist()
sns.distplot(df["UrbanPop"])
df["UrbanPop"].skew()
df["UrbanPop"].kurt()
df["UrbanPop"].describe()

df["Rape"].hist()
sns.distplot(df["Rape"])
df["Rape"].kurt()
df["Rape"].skew()
df["Rape"].describe()


# understanding the relationships between all the four variables#

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df, vars=['Murder', 'Assault', 'UrbanPop', 'Rape']) #to check relationship between more than 1 variables
plt.show()

"""# we can find Positive or Negative Relationships
#correlation
#outliers
#Histograms
#Outliers from the above code between all the four variables instead of doing scatter plot"""

correlation_matrix = df.corr()
#Heat Map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

"""Values close to 1 indicate a strong positive correlation.
Values close to -1 indicate a strong negative correlation.
Values close to 0 indicate a weak or no correlation"""

X = df.iloc[:,1:5].values
X.shape

# transformation on the data #
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)

#construction of dendogram
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(X,method = 'complete'))

#====================================================================================================
#AgglomerativeClustering
#forming a group using clusters
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dend = shc.dendrogram(shc.linkage(X, method='complete'))

cluster_agg = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y_agg = cluster_agg.fit_predict(X)

plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=Y_agg, cmap='rainbow')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Agglomerative Clustering')
plt.colorbar(label='Cluster Label')
plt.show()

#====================================================================================================
#performing k means on the same data
#KMeans (Elbow method)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
Y = kmeans.fit_predict(X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

kmeans.inertia_

kresults = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_predict(X)
    kresults.append(kmeans.inertia_)
    
kresults

#Scatterplot
import matplotlib.pyplot as plt
plt.scatter(x=range(1,11),y=kresults)
plt.plot(range(1,11),kresults,color="red")
plt.show()

"""according to the elbow method we can get clarity on upto  which k value should be choosen"""

#====================================================================================================
#DBSCAN
# DBSCAN Clustering
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1.0, min_samples=2)
db.fit(X)

# Scatter plot for DBSCAN Clustering
Y = db.labels_
plt.scatter(X[0], X[1], c=Y, cmap='rainbow')
plt.xlabel('X-axis Label for Feature 1')
plt.ylabel('Y-axis Label for Feature 2')
plt.title('DBSCAN Clustering Scatter Plot')
plt.colorbar(label='Cluster Label')
plt.show()

# The data has been clustered into five groups based on the Euclidean distance and complete linkage method.
# Each point in the scatter plot is colored according to its cluster label.
# The Elbow method is used to determine the optimal number of clusters, which seems to be 5 based on the plot.
# DBSCAN identifies outliers as points with a label of -1.
# The noise points are excluded from the final visualization, and the data is plotted with cluster labels.
# Points in the scatter plot are colored according to their cluster label.
#=======================================================================================

