import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

credits = pd.read_csv("tmdb_5000_credits.csv")

movies_df = pd.read_csv("tmdb_5000_movies.csv")

#print(credits.head())

# to see the columns names of the movies_df
#print(movies_df.columns)

# renaming the column_name of credits from movie_id to id
credits_column_renamed = credits.rename(index=str, columns={"movie_id":"id"})   

# merging the credits and movies by the use of id
movies_df_merge = movies_df.merge(credits_column_renamed, on="id")

# to see the columns names of the movies_df_merge
#print(movies_df_merge.columns)

# to check the number of columns and rows in the merged tables
#print("merge Datframe :",movies_df_merge.shape)

# dropping the unncessary columns
movies_cleaned_df = movies_df_merge.drop(columns =['homepage','original_language','title_x','title_y','status','production_countries'])
#print(movies_cleaned_df.shape)

#print(movies_cleaned_df.head(1)['overview'])

# applying tfidfvecorizer and removing the stop words in order to create document matrix
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern='\w{1,}',ngram_range=(1,3),stop_words='english')

# filling NaN with blank 
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')

# applying fit transform of overiview with tfv doci=ument_matrix
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])
#print(tfv_matrix)

# computing the sigmoid value of the tfv_matrix, this will apply sigmoid funtion with each movies overview with others 
sig = sigmoid_kernel(tfv_matrix,tfv_matrix)
#print(sig[1])

# reverse indexing of movies index with movies names
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()

#print(type(indices))

def give_recommendation(title , sig=sig):
    # getting index of the movie by the title
    idx = indices[title]

    # enumrating the score of the movie from the sigmoidal_kernel
    sig_scores = list(enumerate(sig[idx]))

    # sorting the sigmoidal score in descending order
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # scores of top 15 recommendations by using slicing
    sig_scores=sig_scores[1:16]

    # getting index of each recommendation
    movie_indices = [i[0] for i in sig_scores]

    # top 15 recommendations
    return movies_cleaned_df['original_title'].iloc[movie_indices]

print((give_recommendation('Avengers: Age of Ultron')))







