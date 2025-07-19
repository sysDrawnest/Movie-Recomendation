import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .recommendation-card {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #F8F9FA;
    }
</style>
""", unsafe_allow_html=True)

class MovieRecommendationEngine:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=10, random_state=42)  # Reduced components
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        
    def create_sample_dataset(self):
        """Create a comprehensive sample movie dataset"""
        np.random.seed(42)
        
        # Sample movie data with diverse genres and features
        movies_data = {
            'title': [
                'The Dark Knight', 'Inception', 'The Matrix', 'Pulp Fiction', 'The Godfather',
                'Forrest Gump', 'The Shawshank Redemption', 'Fight Club', 'Goodfellas', 'The Lord of the Rings',
                'Star Wars', 'Titanic', 'Avatar', 'The Avengers', 'Jurassic Park',
                'Casablanca', 'Gone with the Wind', 'The Wizard of Oz', 'Citizen Kane', 'Vertigo',
                'Psycho', 'Singin\' in the Rain', '2001: A Space Odyssey', 'Sunset Boulevard', 'Apocalypse Now',
                'The Terminator', 'Alien', 'Blade Runner', 'Die Hard', 'Jaws',
                'E.T.', 'Back to the Future', 'Raiders of the Lost Ark', 'Rocky', 'Taxi Driver',
                'Scarface', 'The Silence of the Lambs', 'Se7en', 'The Departed', 'Heat',
                'Gladiator', 'Braveheart', 'The Lion King', 'Toy Story', 'Finding Nemo',
                'Shrek', 'The Incredibles', 'WALL-E', 'Up', 'Frozen',
                'La La Land', 'The Greatest Showman', 'A Star is Born', 'Bohemian Rhapsody', 'Rocketman',
                'John Wick', 'Mad Max: Fury Road', 'The Fast and the Furious', 'Mission: Impossible', 'James Bond: Skyfall'
            ],
            'genres': [
                'Action Crime Drama', 'Action Sci-Fi Thriller', 'Action Sci-Fi', 'Crime Drama', 'Crime Drama',
                'Drama Romance', 'Drama', 'Drama Thriller', 'Crime Drama', 'Adventure Fantasy',
                'Adventure Sci-Fi', 'Drama Romance', 'Action Sci-Fi', 'Action Adventure', 'Adventure Sci-Fi Thriller',
                'Drama Romance War', 'Drama Romance War', 'Adventure Family Fantasy', 'Drama Mystery', 'Mystery Thriller',
                'Horror Thriller', 'Comedy Musical Romance', 'Sci-Fi', 'Drama Film-Noir', 'Drama War',
                'Action Sci-Fi Thriller', 'Horror Sci-Fi Thriller', 'Sci-Fi Thriller', 'Action Thriller', 'Adventure Thriller',
                'Family Sci-Fi', 'Adventure Comedy Sci-Fi', 'Action Adventure', 'Drama Sport', 'Crime Drama Thriller',
                'Crime Drama', 'Crime Horror Thriller', 'Crime Mystery Thriller', 'Crime Drama Thriller', 'Action Crime Thriller',
                'Action Drama', 'Biography Drama War', 'Animation Adventure Comedy', 'Animation Adventure Comedy', 'Adventure Comedy Family',
                'Adventure Animation Comedy', 'Animation Adventure Family', 'Animation Adventure Family', 'Animation Adventure Comedy', 'Animation Adventure Family',
                'Comedy Drama Musical', 'Drama Musical', 'Drama Music Romance', 'Biography Drama Music', 'Biography Drama Musical',
                'Action Crime Thriller', 'Action Adventure Sci-Fi', 'Action Crime Thriller', 'Action Adventure Thriller', 'Action Adventure Thriller'
            ],
            'director': [
                'Christopher Nolan', 'Christopher Nolan', 'Lana Wachowski', 'Quentin Tarantino', 'Francis Ford Coppola',
                'Robert Zemeckis', 'Frank Darabont', 'David Fincher', 'Martin Scorsese', 'Peter Jackson',
                'George Lucas', 'James Cameron', 'James Cameron', 'Joss Whedon', 'Steven Spielberg',
                'Michael Curtiz', 'Victor Fleming', 'Victor Fleming', 'Orson Welles', 'Alfred Hitchcock',
                'Alfred Hitchcock', 'Gene Kelly', 'Stanley Kubrick', 'Billy Wilder', 'Francis Ford Coppola',
                'James Cameron', 'Ridley Scott', 'Ridley Scott', 'John McTiernan', 'Steven Spielberg',
                'Steven Spielberg', 'Robert Zemeckis', 'Steven Spielberg', 'John G. Avildsen', 'Martin Scorsese',
                'Brian De Palma', 'Jonathan Demme', 'David Fincher', 'Martin Scorsese', 'Michael Mann',
                'Ridley Scott', 'Mel Gibson', 'Roger Allers', 'John Lasseter', 'Andrew Stanton',
                'Andrew Adamson', 'Brad Bird', 'Andrew Stanton', 'Pete Docter', 'Chris Buck',
                'Damien Chazelle', 'Michael Gracey', 'Bradley Cooper', 'Bryan Singer', 'Dexter Fletcher',
                'Chad Stahelski', 'George Miller', 'Rob Cohen', 'Brian De Palma', 'Sam Mendes'
            ],
            'year': [
                2008, 2010, 1999, 1994, 1972, 1994, 1994, 1999, 1990, 2001,
                1977, 1997, 2009, 2012, 1993, 1942, 1939, 1939, 1941, 1958,
                1960, 1952, 1968, 1950, 1979, 1984, 1979, 1982, 1988, 1975,
                1982, 1985, 1981, 1976, 1976, 1983, 1991, 1995, 2006, 1995,
                2000, 1995, 1994, 1995, 2003, 2001, 2004, 2008, 2009, 2013,
                2016, 2017, 2018, 2018, 2019, 2014, 2015, 2001, 1996, 2012
            ],
            'rating': [
                9.0, 8.8, 8.7, 8.9, 9.2, 8.8, 9.3, 8.8, 8.7, 8.9,
                8.6, 7.8, 7.9, 8.0, 8.1, 8.5, 8.1, 8.0, 8.3, 8.3,
                8.5, 8.3, 8.3, 8.4, 8.4, 8.0, 8.4, 8.1, 8.2, 8.0,
                7.8, 8.5, 8.4, 8.1, 8.3, 8.3, 8.6, 8.6, 8.5, 8.2,
                8.5, 8.3, 8.5, 8.3, 8.2, 7.8, 8.0, 8.4, 8.3, 7.4,
                8.0, 7.5, 7.6, 7.9, 7.3, 7.4, 8.1, 6.8, 7.7, 7.8
            ],
            'duration': [
                152, 148, 136, 154, 175, 142, 142, 139, 146, 178,
                121, 194, 162, 143, 127, 102, 238, 102, 119, 128,
                109, 103, 149, 110, 147, 107, 117, 117, 132, 124,
                115, 116, 115, 120, 114, 170, 118, 127, 151, 170,
                155, 178, 88, 81, 100, 90, 115, 98, 96, 102,
                128, 105, 136, 134, 121, 101, 120, 106, 131, 143
            ]
        }
        
        # Create DataFrame
        df = pd.DataFrame(movies_data)
        
        # Add derived features
        df['decade'] = (df['year'] // 10) * 10
        df['genre_count'] = df['genres'].apply(lambda x: len(x.split()))
        df['is_recent'] = (df['year'] >= 2000).astype(int)
        df['rating_category'] = pd.cut(df['rating'], bins=[0, 7, 8, 10], labels=['Good', 'Great', 'Masterpiece'])
        df['duration_category'] = pd.cut(df['duration'], bins=[0, 100, 150, 300], labels=['Short', 'Medium', 'Long'])
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the movie data for recommendation algorithms"""
        # Create combined features for content-based filtering
        df['combined_features'] = (
            df['genres'] + ' ' + 
            df['director'] + ' ' +
            df['decade'].astype(str) + ' ' +
            df['rating_category'].astype(str) + ' ' +
            df['duration_category'].astype(str)
        )
        
        # Prepare numerical features for KNN
        numerical_features = ['year', 'rating', 'duration', 'genre_count', 'is_recent']
        df_numerical = df[numerical_features].copy()
        
        return df, df_numerical
    
    def build_tfidf_recommender(self, df):
        """Build TF-IDF based recommender"""
        tfidf = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
        self.tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        
    def build_knn_recommender(self, df_numerical):
        """Build KNN based recommender"""
        scaled_features = self.scaler.fit_transform(df_numerical)
        self.knn_model.fit(scaled_features)
        
    def build_svd_recommender(self, df):
        """Build SVD based recommender"""
        # Create genre matrix for SVD
        genres = []
        for genre_list in df['genres']:
            genres.extend(genre_list.split())
        unique_genres = list(set(genres))
        
        # Create binary genre matrix
        genre_matrix = np.zeros((len(df), len(unique_genres)))
        for i, genre_list in enumerate(df['genres']):
            for genre in genre_list.split():
                if genre in unique_genres:
                    genre_matrix[i, unique_genres.index(genre)] = 1
        
        # Ensure n_components doesn't exceed n_features
        n_components = min(10, len(unique_genres) - 1, len(df) - 1)
        if n_components < 1:
            n_components = 1
            
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Apply SVD
        try:
            self.svd_matrix = self.svd.fit_transform(genre_matrix)
        except ValueError as e:
            st.warning(f"SVD initialization issue: {e}. Using reduced components.")
            self.svd = TruncatedSVD(n_components=min(3, len(unique_genres)//2), random_state=42)
            self.svd_matrix = self.svd.fit_transform(genre_matrix)
        
    def get_tfidf_recommendations(self, movie_title, n_recommendations=10):
        """Get recommendations using TF-IDF similarity"""
        try:
            idx = self.df[self.df['title'] == movie_title].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations+1]  # Exclude the movie itself
            movie_indices = [i[0] for i in sim_scores]
            
            recommendations = self.df.iloc[movie_indices][['title', 'genres', 'director', 'year', 'rating']].copy()
            recommendations['similarity_score'] = [score[1] for score in sim_scores]
            
            return recommendations
        except IndexError:
            return pd.DataFrame()
    
    def get_knn_recommendations(self, movie_title, n_recommendations=10):
        """Get recommendations using KNN"""
        try:
            idx = self.df[self.df['title'] == movie_title].index[0]
            movie_features = self.scaler.transform([self.df_numerical.iloc[idx]])
            
            distances, indices = self.knn_model.kneighbors(movie_features, n_neighbors=n_recommendations+1)
            indices = indices[0][1:]  # Exclude the movie itself
            
            recommendations = self.df.iloc[indices][['title', 'genres', 'director', 'year', 'rating']].copy()
            recommendations['similarity_score'] = 1 - distances[0][1:n_recommendations+1]
            
            return recommendations
        except (IndexError, ValueError):
            return pd.DataFrame()
    
    def get_svd_recommendations(self, movie_title, n_recommendations=10):
        """Get recommendations using SVD"""
        try:
            idx = self.df[self.df['title'] == movie_title].index[0]
            movie_vector = self.svd_matrix[idx].reshape(1, -1)
            
            # Calculate similarity with all movies
            similarities = cosine_similarity(movie_vector, self.svd_matrix)[0]
            sim_indices = similarities.argsort()[::-1][1:n_recommendations+1]  # Exclude the movie itself
            
            recommendations = self.df.iloc[sim_indices][['title', 'genres', 'director', 'year', 'rating']].copy()
            recommendations['similarity_score'] = similarities[sim_indices]
            
            return recommendations
        except (IndexError, AttributeError) as e:
            st.warning(f"SVD recommendations unavailable: {e}")
            return pd.DataFrame()
    
    def get_hybrid_recommendations(self, movie_title, n_recommendations=10):
        """Get hybrid recommendations combining all methods"""
        tfidf_recs = self.get_tfidf_recommendations(movie_title, n_recommendations)
        knn_recs = self.get_knn_recommendations(movie_title, n_recommendations)
        svd_recs = self.get_svd_recommendations(movie_title, n_recommendations)
        
        # Combine recommendations with weighted scoring
        all_recs = []
        
        for df_rec, weight in [(tfidf_recs, 0.4), (knn_recs, 0.3), (svd_recs, 0.3)]:
            if not df_rec.empty:
                df_rec = df_rec.copy()
                df_rec['weighted_score'] = df_rec['similarity_score'] * weight
                all_recs.append(df_rec)
        
        if not all_recs:
            return pd.DataFrame()
        
        # Combine and aggregate scores
        combined = pd.concat(all_recs)
        hybrid_recs = combined.groupby('title').agg({
            'genres': 'first',
            'director': 'first',
            'year': 'first',
            'rating': 'first',
            'weighted_score': 'sum'
        }).reset_index()
        
        hybrid_recs = hybrid_recs.sort_values('weighted_score', ascending=False).head(n_recommendations)
        
        return hybrid_recs
    
    def get_popular_recommendations(self, genre_filter=None, n_recommendations=10):
        """Get popular movie recommendations"""
        df_filtered = self.df.copy()
        
        if genre_filter and genre_filter != "All":
            df_filtered = df_filtered[df_filtered['genres'].str.contains(genre_filter, case=False)]
        
        popular_movies = df_filtered.nlargest(n_recommendations, 'rating')[['title', 'genres', 'director', 'year', 'rating']]
        
        return popular_movies

def create_visualizations(df):
    """Create various visualizations for the dataset"""
    
    # 1. Rating distribution
    fig_rating = px.histogram(
        df, x='rating', nbins=20, title='Movie Rating Distribution',
        color_discrete_sequence=['#FF6B6B']
    )
    fig_rating.update_layout(showlegend=False)
    
    # 2. Movies by decade
    decade_counts = df['decade'].value_counts().sort_index()
    fig_decade = px.bar(
        x=decade_counts.index, y=decade_counts.values,
        title='Movies by Decade', labels={'x': 'Decade', 'y': 'Number of Movies'},
        color_discrete_sequence=['#4ECDC4']
    )
    
    # 3. Genre popularity
    all_genres = []
    for genres in df['genres']:
        all_genres.extend(genres.split())
    genre_counts = Counter(all_genres).most_common(10)
    
    fig_genre = px.bar(
        x=[g[1] for g in genre_counts], y=[g[0] for g in genre_counts],
        orientation='h', title='Top 10 Most Popular Genres',
        labels={'x': 'Count', 'y': 'Genre'},
        color_discrete_sequence=['#45B7D1']
    )
    
    # 4. Rating vs Year scatter
    fig_scatter = px.scatter(
        df, x='year', y='rating', size='duration',
        hover_data=['title', 'director'], title='Movie Ratings Over Time',
        color='rating', color_continuous_scale='Viridis'
    )
    
    return fig_rating, fig_decade, fig_genre, fig_scatter

def main():
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation Engine</h1>', unsafe_allow_html=True)
    
    # Initialize the recommendation engine
    if 'recommender' not in st.session_state:
        with st.spinner('Initializing recommendation engine...'):
            st.session_state.recommender = MovieRecommendationEngine()
            st.session_state.recommender.df = st.session_state.recommender.create_sample_dataset()
            st.session_state.recommender.df, st.session_state.recommender.df_numerical = st.session_state.recommender.preprocess_data(st.session_state.recommender.df)
            
            # Build all recommendation models
            st.session_state.recommender.build_tfidf_recommender(st.session_state.recommender.df)
            st.session_state.recommender.build_knn_recommender(st.session_state.recommender.df_numerical)
            st.session_state.recommender.build_svd_recommender(st.session_state.recommender.df)
    
    recommender = st.session_state.recommender
    df = recommender.df
    
    # Sidebar for navigation and controls
    st.sidebar.markdown("## 🎮 Control Panel")
    
    page = st.sidebar.selectbox(
        "Choose Page",
        ["🏠 Home", "🔍 Movie Recommendations", "📊 Dataset Analysis", "📈 Visualizations"]
    )
    
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Movies", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Rating", f"{df['rating'].mean():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            unique_genres = set()
            for genres in df['genres']:
                unique_genres.update(genres.split())
            st.metric("Unique Genres", len(unique_genres))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ## 🎯 About This Movie Recommendation Engine
        
        This comprehensive movie recommendation system uses multiple machine learning algorithms to provide personalized movie suggestions:
        
        ### 🔧 **Algorithms Used:**
        - **TF-IDF Vectorization**: Analyzes movie content similarity based on genres, directors, and metadata
        - **K-Nearest Neighbors (KNN)**: Finds similar movies based on numerical features
        - **Singular Value Decomposition (SVD)**: Reduces dimensionality for genre-based recommendations
        - **Hybrid Approach**: Combines all methods for more accurate recommendations
        
        ### ✨ **Features:**
        - Multiple recommendation algorithms
        - Interactive visualizations
        - Dataset analysis and insights
        - Popular movie recommendations
        - Comprehensive movie database
        
        ### 🚀 **How to Use:**
        1. Go to the "Movie Recommendations" page
        2. Select a movie you like
        3. Choose your preferred algorithm
        4. Get personalized recommendations!
        """)
        
        # Recent additions
        st.markdown("### 🆕 Recently Added Movies")
        recent_movies = df.nlargest(5, 'year')[['title', 'year', 'rating', 'genres']]
        st.dataframe(recent_movies, use_container_width=True)
    
    elif page == "🔍 Movie Recommendations":
        st.markdown("## 🎬 Get Movie Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_movie = st.selectbox(
                "Select a movie you enjoyed:",
                options=df['title'].tolist(),
                help="Choose a movie to get similar recommendations"
            )
        
        with col2:
            algorithm = st.selectbox(
                "Choose Algorithm:",
                ["Hybrid (Recommended)", "TF-IDF Content", "KNN Similarity", "SVD Genre-based"]
            )
        
        num_recommendations = st.slider("Number of recommendations:", 5, 15, 10)
        
        if st.button("🎯 Get Recommendations", type="primary"):
            with st.spinner(f'Finding movies similar to "{selected_movie}"...'):
                
                if algorithm == "Hybrid (Recommended)":
                    recommendations = recommender.get_hybrid_recommendations(selected_movie, num_recommendations)
                    score_col = 'weighted_score'
                elif algorithm == "TF-IDF Content":
                    recommendations = recommender.get_tfidf_recommendations(selected_movie, num_recommendations)
                    score_col = 'similarity_score'
                elif algorithm == "KNN Similarity":
                    recommendations = recommender.get_knn_recommendations(selected_movie, num_recommendations)
                    score_col = 'similarity_score'
                else:  # SVD
                    recommendations = recommender.get_svd_recommendations(selected_movie, num_recommendations)
                    score_col = 'similarity_score'
                
                if not recommendations.empty:
                    st.markdown(f"### 🎬 Movies Similar to '{selected_movie}'")
                    
                    for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{idx}. {movie['title']} ({movie['year']})</h4>
                            <p><strong>Director:</strong> {movie['director']}</p>
                            <p><strong>Genres:</strong> {movie['genres']}</p>
                            <p><strong>Rating:</strong> ⭐ {movie['rating']}/10</p>
                            <p><strong>Similarity Score:</strong> {movie[score_col]:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("No recommendations found. Please try a different movie.")
        
        st.markdown("---")
        
        # Popular recommendations section
        st.markdown("### 🔥 Popular Movies")
        
        genre_filter = st.selectbox(
            "Filter by genre:",
            ["All"] + sorted(['Action', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi', 'Romance', 'Adventure', 'Crime'])
        )
        
        popular_movies = recommender.get_popular_recommendations(genre_filter, 10)
        
        if not popular_movies.empty:
            for idx, (_, movie) in enumerate(popular_movies.iterrows(), 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{idx}. {movie['title']}** ({movie['year']}) - {movie['director']}")
                    st.write(f"*{movie['genres']}*")
                with col2:
                    st.metric("Rating", f"{movie['rating']}/10")
    
    elif page == "📊 Dataset Analysis":
        st.markdown("## 📊 Dataset Analysis")
        
        # Dataset overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Dataset Overview")
            st.write(f"**Total Movies:** {len(df)}")
            st.write(f"**Year Range:** {df['year'].min()} - {df['year'].max()}")
            st.write(f"**Average Rating:** {df['rating'].mean():.2f}")
            st.write(f"**Average Duration:** {df['duration'].mean():.0f} minutes")
        
        with col2:
            st.markdown("### 🏆 Top Statistics")
            st.write(f"**Highest Rated:** {df.loc[df['rating'].idxmax(), 'title']} ({df['rating'].max()})")
            st.write(f"**Longest Movie:** {df.loc[df['duration'].idxmax(), 'title']} ({df['duration'].max()} min)")
            st.write(f"**Most Recent:** {df.loc[df['year'].idxmax(), 'title']} ({df['year'].max()})")
            st.write(f"**Oldest Movie:** {df.loc[df['year'].idxmin(), 'title']} ({df['year'].min()})")
        
        st.markdown("---")
        
        # Detailed statistics
        st.markdown("### 📈 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Rating Statistics**")
            st.dataframe(df['rating'].describe())
        
        with col2:
            st.markdown("**Duration Statistics**")
            st.dataframe(df['duration'].describe())
        
        # Top directors and genres
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎬 Top Directors")
            director_counts = df['director'].value_counts().head(10)
            st.dataframe(director_counts)
        
        with col2:
            st.markdown("### 🎭 Genre Analysis")
            all_genres = []
            for genres in df['genres']:
                all_genres.extend(genres.split())
            genre_counts = Counter(all_genres).most_common(10)
            genre_df = pd.DataFrame(genre_counts, columns=['Genre', 'Count'])
            st.dataframe(genre_df)
        
        # Full dataset view
        st.markdown("### 📋 Complete Dataset")
        st.dataframe(df, use_container_width=True)
    
    elif page == "📈 Visualizations":
        st.markdown("## 📈 Data Visualizations")
        
        # Create visualizations
        fig_rating, fig_decade, fig_genre, fig_scatter = create_visualizations(df)
        
        # Display visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_rating, use_container_width=True)
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_decade, use_container_width=True)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Additional insights
        st.markdown("### 🔍 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_rating_by_decade = df.groupby('decade')['rating'].mean()
            best_decade = avg_rating_by_decade.idxmax()
            st.metric(
                "Best Decade (Avg Rating)", 
                f"{best_decade}s", 
                f"{avg_rating_by_decade[best_decade]:.2f}"
            )
        
        with col2:
            most_common_duration = df['duration_category'].value_counts().idxmax()
            st.metric("Most Common Duration", most_common_duration, 
                     f"{df['duration_category'].value_counts()[most_common_duration]} movies")
        
        with col3:
            correlation = df['year'].corr(df['rating'])
            trend = "↗️" if correlation > 0 else "↘️" if correlation < 0 else "➡️"
            st.metric("Rating Trend Over Time", f"{trend} {correlation:.3f}", 
                     "Positive correlation" if correlation > 0 else "Negative correlation")
        
        # Genre co-occurrence analysis
        st.markdown("### 🎭 Genre Analysis")
        
        # Create genre co-occurrence matrix
        all_genres = set()
        for genres in df['genres']:
            all_genres.update(genres.split())
        
        genre_list = sorted(list(all_genres))
        co_occurrence = np.zeros((len(genre_list), len(genre_list)))
        
        for genres in df['genres']:
            genre_split = genres.split()
            for i, genre1 in enumerate(genre_list):
                for j, genre2 in enumerate(genre_list):
                    if genre1 in genre_split and genre2 in genre_split:
                        co_occurrence[i][j] += 1
        
        # Create heatmap
        fig_heatmap = px.imshow(
            co_occurrence,
            labels=dict(x="Genre", y="Genre", color="Co-occurrence"),
            x=genre_list,
            y=genre_list,
            title="Genre Co-occurrence Heatmap",
            color_continuous_scale="Blues"
        )
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Word cloud for genres
        st.markdown("### ☁️ Genre Word Cloud")
        
        all_genres_text = ' '.join([genre for genres in df['genres'] for genre in genres.split()])
        
        if all_genres_text:
            wordcloud = WordCloud(
                width=800, height=400, 
                background_color='white',
                colormap='viridis'
            ).generate(all_genres_text)
            
            fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.title('Most Popular Genres', fontsize=16, fontweight='bold')
            st.pyplot(fig_wordcloud)
        
        # Performance comparison of algorithms
        st.markdown("### ⚡ Algorithm Performance Comparison")
        
        st.info("""
        **Algorithm Comparison:**
        
        - **TF-IDF Content-Based**: Best for finding movies with similar themes, genres, and directors
        - **KNN Similarity**: Excellent for finding movies with similar numerical characteristics (rating, year, duration)
        - **SVD Genre-Based**: Great for discovering movies within similar genre spaces
        - **Hybrid Approach**: Combines all methods for the most comprehensive recommendations
        
        **Recommendation**: Use the Hybrid approach for best results, or try different algorithms to explore various aspects of similarity!
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎬 Movie Recommendation Engine | Built with Streamlit, Scikit-learn & ❤️</p>
        <p>Featuring TF-IDF, KNN, SVD & Hybrid Algorithms</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
