import streamlit as st
import pickle
import pandas as pd

def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_coll=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    rec_movies=[]
    for i in movies_coll:
        movie_id=i[0]
        #fetching movie poster
        
        rec_movies.append(movies.iloc[i[0]].title)
    return rec_movies

movies_list=pickle.load(open('movies.pkl','rb'))
movies=pd.DataFrame(movies_list)

similarity=pickle.load(open('similarity.pkl','rb'))
st.title('Movie Recommender')

selected = st.selectbox(
    'Select a movie...',
    movies['title'].values)

if st.button('Recommend'):
    recommendations=recommend(selected)
    for i in recommendations:
        st.write(i)
