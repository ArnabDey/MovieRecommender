import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import csv

movies = pd.read_csv('./ml-latest-small/movies.csv')
tags  = pd.read_csv('./ml-latest-small/tags.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# Organizing the Data
averageRatings = ratings.groupby(['movieId'], as_index = False).mean()
combinedTags = tags.groupby('movieId')['tag'].apply(list).reset_index()
movieRating = pd.merge(averageRatings, movies, on = 'movieId', how = 'outer').fillna('')
combinedData = pd.merge(movieRating, combinedTags, on = 'movieId', how = 'outer').fillna('')

combinedData['genres'] = combinedData['genres'].str.split('|')

# Use One-Hot-Encoding for the genres of the movie
combinedData2 = combinedData.drop('genres', 1).join(combinedData.genres.str.join('|').str.get_dummies())

# Generate new csv file after pre-processing the data
combinedData2.to_csv('mergedata.csv', index = False)


combinedData2=combinedData2.fillna(combinedData2.mean())

# Determine number of clusters for kmenas
from sklearn.cluster import KMeans
wccss = []
formattedData = pd.concat([combinedData2.iloc[:,2:3], combinedData2.iloc[:,6:26]], axis = 1)
formattedData = formattedData.replace(r'^\s*$', np.nan, regex=True)
formattedData = formattedData.fillna(formattedData.mean())
X = formattedData.values
colNames = list(combinedData2.columns.values)[6:]
for i in range(1, 31):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 30, random_state = 0)
     kmeans.fit(X)
     wccss.append(kmeans.inertia_)
print(len(wccss))
plt.plot(range(1, 31), wccss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCCS')
plt.show()

# KMeans Model
kmeans = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 300, n_init = 15, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the Clusters
# Separate Graphs for Each Genere of Movie 
for i in range(1, 21):
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, i], s = 20, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, i], s = 20, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, i], s = 20, c = 'orange', label = 'Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, i], s = 20, c = 'yellow', label = 'Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, i], s = 20, c = 'green', label = 'Cluster 5')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 20, c = 'purple', label = 'Cluster 6')
    plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 20, c = 'black', label = 'Cluster 7')
    plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 20, c = 'pink', label = 'Cluster 8')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'cyan', label = 'Centroids')
    plt.title('Clusters of movie type: ' + colNames[i - 1])
    plt.xlabel('Movie Rating (1-5)')
    plt.ylabel('Movie Genre Score')
    plt.legend()
    plt.show()
    
# Principal Component Analysis to generate one Graph to visualize clustering
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# KMeans Model for PCA
kmeans = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 300, n_init = 15, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'orange', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 20, c = 'green', label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 20, c = 'purple', label = 'Cluster 6')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 20, c = 'black', label = 'Cluster 7')
plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 20, c = 'pink', label = 'Cluster 8')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'cyan', label = 'Centroids')
plt.title('Clustering for all movie type')
plt.xlabel('Movie Rating (1-5)')
plt.ylabel('Movie Genre Score')
plt.legend()
plt.show()

# Predictions can be made using this model

