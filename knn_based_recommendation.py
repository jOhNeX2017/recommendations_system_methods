import numpy as np
import pandas as pd
from scipy.sparse import  csr_matrix
from sklearn.neighbors import NearestNeighbors

movies_df = pd.read_csv('movies_lens_dataset/movies.csv',usecols=['movieId','title'],dtype={'movieId':'int32','title':'string'})
rating_df = pd.read_csv('movies_lens_dataset/ratings.csv',usecols=['userId','movieId','rating'],dtype={'userId':'int32','movieId':'int32','rating':'float32'})

#print(movies_df.info())

# merging both the csv wrt movieID
df=pd.merge(movies_df,rating_df,on="movieId")

# combining movie with its rating and counting total rating given by each users
combine_movie_rating=df.dropna(axis=0 , subset=['title'])
movie_rating_count = combine_movie_rating.groupby(by=['title'])['rating'].count().reset_index()[['title','rating']]

# renaming rating columns to totalRatingCount
movie_rating_count=movie_rating_count.rename(columns={'rating':'totalRatingCount'})

#print(movie_rating_count.head())

# rating of each movie owth total count of rating 
rating_with_totalRatingCount = pd.merge(combine_movie_rating,movie_rating_count,left_on="title",right_on="title",how="left")
#print(rating_with_totalRatingCount.head())

pd.set_option('display.float_format', lambda x: '%3f' %x)
#print(movie_rating_count['totalRatingCount'].describe())

# setting the threshold value and discaring all the movies which have rating count equalt to maore than threshold
popularity_threshold = 50
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
#print(rating_popular_movie.head())

# creating the pivot table & filling na with 0 value
movie_features_df = rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)
#print(movie_features_df.head())

# now crating the pivate matrix from the pivot table
movie_features_df_matrix = csr_matrix(movie_features_df.values)

# using nearest neighbours 
knn_model = NearestNeighbors(metric='cosine', algorithm = 'brute')
knn_model.fit(movie_features_df_matrix)

query_index = np.random.choice(movie_features_df.shape[1])
distance,indices = knn_model.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)

for i in range(0,len(distance.flatten())):
    if i == 0 :
        print('Recommedation for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with disatnce of {2}'.format(i,movie_features_df.index[indices.flatten()[i]],distance.flatten()[i]))