import numpy as np
import pandas as pd
import ast


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')
#print(movies.head())
#merge the csv files
#movies.merge(credits,on='title').shape ->4809,23
movies=movies.merge(credits,on='title')

#=======DATA PROCESSING=========
#removing useless columns, reqd columns-genre,id,keywords,title,overview,cast,crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

#missing data
movies.dropna(inplace=True)
#print(movies.isnull().sum())

#format data
def convert(obj):
    list=[]
    for i in ast.literal_eval(obj):
        list.append(i['name'])
    return list

movies['genres']=movies['genres'].apply(convert)
#print(movies.genres)
movies['keywords']=movies['keywords'].apply(convert)
#print(movies.keywords)

def convert_cast(obj): #formatting cast upto 3 actors
    list=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            list.append(i['name'])
            counter+=1
        else:
            break
    return list
movies['cast']=movies['cast'].apply(convert_cast)
#print(movies.cast)

def get_director(obj): #keeping only director names from crew column
    list=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            list.append(i['name'])
            break
    return list
movies['crew']=movies['crew'].apply(get_director)
#print(movies.crew)

movies['overview']=movies['overview'].apply(lambda x:x.split()) #string->list
#print(movies.overview)

#removing white spaces in data values
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
#print(movies.genres)
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

#combining columns into tags
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
#print(movies.tags)

#creating new dataframe
new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x)) #converting to string
new_df['tags']=new_df['tags'].apply(lambda x: x.lower())

#=======VECTORIZATION=========
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer


#print(cv.get_feature_names())->5000 frequent words
ps=PorterStemmer() #stemming
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags']=new_df['tags'].apply(stem)

cv=CountVectorizer(max_features=5000,stop_words='english') #getting 5000 frequestly used words excluding stop words
vectors=cv.fit_transform(new_df['tags']).toarray()
#print(cv.get_feature_names())
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman')

import pickle
pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))