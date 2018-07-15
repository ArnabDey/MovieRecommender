import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import csv

movies = pd.read_csv('./ml-latest-small/movies.csv')
tags  = pd.read_csv('./ml-latest-small/tags.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# Finds sentiment for the movie using NLP
def findSentiment():
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    # Cleaning the dataset
    print(combinedData)

# Organizing the Data
averageRatings = ratings.groupby(['movieId'], as_index = False).mean()
combinedTags = tags.groupby('movieId')['tag'].apply(list).reset_index()
movieRating = pd.merge(averageRatings, movies, on = "movieId", how = "outer").fillna("")
combinedData = pd.merge(movieRating, combinedTags, on = "movieId", how = "outer").fillna("")


combinedData['genres'] = combinedData['genres'].str.split("|")

# Use One-Hot-Encoding for the genres of the movie
combinedData2 = combinedData.drop('genres', 1).join(combinedData.genres.str.join('|').str.get_dummies())

# Generate new csv file after pre-processing the data
combinedData2.to_csv('mergedata.csv', index = False)


combinedData2=combinedData2.fillna(combinedData2.mean())

# Determine number of clusters for kmenas
from sklearn.cluster import KMeans
wccss = []
X = pd.concat([combinedData2.iloc[:,2:3], combinedData2.iloc[:,6:26]], axis = 1)
X = X.replace(r'^\s*$', np.nan, regex=True)
X = X.fillna(X.mean())


for i in range(1, 20):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, random_state = 42)
     kmeans.fit(X)
     wccss.append(kmeans.inertia_)
print(len(wccss))
plt.plot(range(1, 20), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCCS')
plt.show()



