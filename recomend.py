import numpy as np
import sklearn
import pandas as pd
import matplotlib
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
    print(tags)

# Organizing the Data
averageRatings = ratings.groupby(['movieId'], as_index = False).mean()
combinedTags = tags.groupby('movieId')['tag'].apply(list).reset_index()
movieRating = pd.merge(averageRatings, movies, on = "movieId", how = "outer").fillna("")

combinedData = pd.merge(movieRating, combinedTags, on = "movieId", how = "outer").fillna("")
combinedData.to_csv('mergedata.csv', index = False)
