import numpy as np
import sklearn
import pandas as pd
import matplotlib

movies = pd.read_csv('./ml-latest-small/movies.csv')
tags  = pd.read_csv('./ml-latest-small/tags.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

def findById(movieId):
    for index, row in movies.iterrows():
        if (row['movieId'] == movieId):
            return row['title']

def findRatings(movieId):
    ratingArr = []
    for index, row in ratings.iterrows():
        if (row['movieId'] == movieId):
            ratingArr.append(row['rating'])
    sum = 0
    for val in ratingArr:
        sum = sum + val
    return sum * 1.0 / len(ratingArr)

print(findById(2))
print(str(findRatings(2)/ 5 * 100)  + '% precent of people liked the movie')