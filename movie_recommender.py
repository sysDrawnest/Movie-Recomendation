import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on movie ID
movies = movies.merge(credits, left_on='id', right_on='movie_id')

# Keep relevant columns
movies = movies[['title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.rename(columns={'title_x': 'title'}, inplace=True)

# Helper functions to extract director and convert to list
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])  # top 3 actors
movies['crew'] = movies['crew'].apply(get_director)

# Combine features into a single string
movies['tags'] = movies['overview'].fillna('') + ' ' + \
                 movies['genres'].apply(lambda x: " ".join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: " ".join(x)) + ' ' + \
                 movies['cast'].apply(lambda x: " ".join(x)) + ' ' + \
                 movies['crew']

# Convert to lowercase
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

#acces poster
import requests

API_KEY = ""

def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    
    if data['results']:
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return full_path
    return "https://via.placeholder.com/500x750?text=No+Image"


# Recommend function
def recommend(movie):
    if movie not in movies['title'].values:
        return [], []
    idx = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    movie_indices = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_titles = []
    recommended_posters = []

    for i in movie_indices:
        title = movies.iloc[i[0]].title
        recommended_titles.append(title)
        recommended_posters.append(fetch_poster(title))

    return recommended_titles, recommended_posters


# Streamlit app
st.title("SYS Movie Recommendation System")
selected_movie = st.selectbox("Type or select a movie from the dropdown:", movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie)
    st.subheader("Top 5 Recommendations:")

    cols = st.columns(5)
    for idx in range(5):
        with cols[idx]:
            st.image(posters[idx])
            st.caption(names[idx])

