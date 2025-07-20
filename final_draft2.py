import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ast
import requests
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SYS Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        background: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and processing
@st.cache_data
def load_and_process_data():
    """Load and process movie data with error handling"""
    try:
        # Try to load the CSV files
        movies = pd.read_csv('movies.csv')
        credits = pd.read_csv('credits.csv')
        
        # Merge datasets on movie ID
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
        
        # Keep relevant columns, including vote_average and release_date
        required_columns = ['title_x', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'release_date']
        movies = movies[required_columns]
        movies.rename(columns={'title_x': 'title'}, inplace=True)
        
        return movies, True
    except FileNotFoundError:
        st.error("CSV files not found! Please ensure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in your directory.")
        return create_sample_data(), False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data(), False

def create_sample_data():
    """Create sample movie data for demonstration"""
    sample_movies = {
        'title': [
            'The Dark Knight', 'Inception', 'The Matrix', 'Pulp Fiction', 'The Godfather',
            'Forrest Gump', 'The Shawshank Redemption', 'Fight Club', 'Goodfellas', 'The Lord of the Rings',
            'Star Wars', 'Titanic', 'Avatar', 'The Avengers', 'Jurassic Park',
            'Casablanca', 'Gone with the Wind', 'The Wizard of Oz', 'Citizen Kane', 'Vertigo'
        ],
        'overview': [
            'Batman battles the Joker in Gotham City',
            'A thief enters people\'s dreams to steal secrets',
            'A computer hacker discovers reality is a simulation',
            'Interconnected stories of Los Angeles criminals',
            'The aging patriarch of a crime dynasty transfers control',
            'Life story of a simple man with low IQ',
            'Two imprisoned men bond over years finding solace',
            'An insomniac office worker forms an underground fight club',
            'Story of Henry Hill and his life in the mob',
            'A hobbit journeys to destroy a powerful ring',
            'Rebels battle the evil Galactic Empire',
            'A seventeen-year-old aristocrat falls in love with a poor artist',
            'Marines fight alien creatures on planet Pandora',
            'Superheroes assemble to save the world',
            'Scientists clone dinosaurs for a theme park',
            'A cynical American expatriate meets a former lover',
            'Epic historical romance during American Civil War',
            'A young girl travels to a magical land',
            'A publishing tycoon\'s rise and fall',
            'A detective becomes obsessed with a mysterious woman'
        ],
        'genres': [
            '[{"name": "Action"}, {"name": "Crime"}, {"name": "Drama"}]',
            '[{"name": "Action"}, {"name": "Sci-Fi"}, {"name": "Thriller"}]',
            '[{"name": "Action"}, {"name": "Sci-Fi"}]',
            '[{"name": "Crime"}, {"name": "Drama"}]',
            '[{"name": "Crime"}, {"name": "Drama"}]',
            '[{"name": "Drama"}, {"name": "Romance"}]',
            '[{"name": "Drama"}]',
            '[{"name": "Drama"}, {"name": "Thriller"}]',
            '[{"name": "Crime"}, {"name": "Drama"}]',
            '[{"name": "Adventure"}, {"name": "Fantasy"}]',
            '[{"name": "Adventure"}, {"name": "Sci-Fi"}]',
            '[{"name": "Drama"}, {"name": "Romance"}]',
            '[{"name": "Action"}, {"name": "Sci-Fi"}]',
            '[{"name": "Action"}, {"name": "Adventure"}]',
            '[{"name": "Adventure"}, {"name": "Sci-Fi"}]',
            '[{"name": "Drama"}, {"name": "Romance"}]',
            '[{"name": "Drama"}, {"name": "Romance"}]',
            '[{"name": "Adventure"}, {"name": "Family"}]',
            '[{"name": "Drama"}, {"name": "Mystery"}]',
            '[{"name": "Mystery"}, {"name": "Thriller"}]'
        ],
        'keywords': [
            '[{"name": "batman"}, {"name": "joker"}, {"name": "gotham"}]',
            '[{"name": "dreams"}, {"name": "heist"}, {"name": "mind"}]',
            '[{"name": "virtual reality"}, {"name": "hacker"}, {"name": "simulation"}]',
            '[{"name": "gangster"}, {"name": "nonlinear"}, {"name": "los angeles"}]',
            '[{"name": "mafia"}, {"name": "family"}, {"name": "power"}]',
            '[{"name": "vietnam war"}, {"name": "friendship"}, {"name": "life"}]',
            '[{"name": "prison"}, {"name": "friendship"}, {"name": "hope"}]',
            '[{"name": "insomnia"}, {"name": "club"}, {"name": "violence"}]',
            '[{"name": "mafia"}, {"name": "biography"}, {"name": "violence"}]',
            '[{"name": "ring"}, {"name": "wizard"}, {"name": "quest"}]',
            '[{"name": "space"}, {"name": "rebellion"}, {"name": "empire"}]',
            '[{"name": "ship"}, {"name": "disaster"}, {"name": "love"}]',
            '[{"name": "alien"}, {"name": "planet"}, {"name": "military"}]',
            '[{"name": "superhero"}, {"name": "team"}, {"name": "alien invasion"}]',
            '[{"name": "dinosaur"}, {"name": "cloning"}, {"name": "island"}]',
            '[{"name": "world war ii"}, {"name": "bar"}, {"name": "resistance"}]',
            '[{"name": "civil war"}, {"name": "plantation"}, {"name": "survival"}]',
            '[{"name": "tornado"}, {"name": "magic"}, {"name": "home"}]',
            '[{"name": "newspaper"}, {"name": "power"}, {"name": "mystery"}]',
            '[{"name": "obsession"}, {"name": "detective"}, {"name": "fear"}]'
        ],
        'cast': [
            '[{"name": "Christian Bale"}, {"name": "Heath Ledger"}, {"name": "Aaron Eckhart"}]',
            '[{"name": "Leonardo DiCaprio"}, {"name": "Marion Cotillard"}, {"name": "Tom Hardy"}]',
            '[{"name": "Keanu Reeves"}, {"name": "Laurence Fishburne"}, {"name": "Carrie-Anne Moss"}]',
            '[{"name": "John Travolta"}, {"name": "Uma Thurman"}, {"name": "Samuel L. Jackson"}]',
            '[{"name": "Marlon Brando"}, {"name": "Al Pacino"}, {"name": "James Caan"}]',
            '[{"name": "Tom Hanks"}, {"name": "Robin Wright"}, {"name": "Gary Sinise"}]',
            '[{"name": "Tim Robbins"}, {"name": "Morgan Freeman"}, {"name": "Bob Gunton"}]',
            '[{"name": "Brad Pitt"}, {"name": "Edward Norton"}, {"name": "Helena Bonham Carter"}]',
            '[{"name": "Robert De Niro"}, {"name": "Ray Liotta"}, {"name": "Joe Pesci"}]',
            '[{"name": "Elijah Wood"}, {"name": "Ian McKellen"}, {"name": "Orlando Bloom"}]',
            '[{"name": "Mark Hamill"}, {"name": "Harrison Ford"}, {"name": "Carrie Fisher"}]',
            '[{"name": "Leonardo DiCaprio"}, {"name": "Kate Winslet"}, {"name": "Billy Zane"}]',
            '[{"name": "Sam Worthington"}, {"name": "Zoe Saldana"}, {"name": "Sigourney Weaver"}]',
            '[{"name": "Robert Downey Jr."}, {"name": "Chris Evans"}, {"name": "Mark Ruffalo"}]',
            '[{"name": "Sam Neill"}, {"name": "Laura Dern"}, {"name": "Jeff Goldblum"}]',
            '[{"name": "Humphrey Bogart"}, {"name": "Ingrid Bergman"}, {"name": "Paul Henreid"}]',
            '[{"name": "Clark Gable"}, {"name": "Vivien Leigh"}, {"name": "Thomas Mitchell"}]',
            '[{"name": "Judy Garland"}, {"name": "Frank Morgan"}, {"name": "Ray Bolger"}]',
            '[{"name": "Orson Welles"}, {"name": "Joseph Cotten"}, {"name": "Dorothy Comingore"}]',
            '[{"name": "James Stewart"}, {"name": "Kim Novak"}, {"name": "Barbara Bel Geddes"}]'
        ],
        'crew': [
            '[{"name": "Christopher Nolan", "job": "Director"}]',
            '[{"name": "Christopher Nolan", "job": "Director"}]',
            '[{"name": "Lana Wachowski", "job": "Director"}]',
            '[{"name": "Quentin Tarantino", "job": "Director"}]',
            '[{"name": "Francis Ford Coppola", "job": "Director"}]',
            '[{"name": "Robert Zemeckis", "job": "Director"}]',
            '[{"name": "Frank Darabont", "job": "Director"}]',
            '[{"name": "David Fincher", "job": "Director"}]',
            '[{"name": "Martin Scorsese", "job": "Director"}]',
            '[{"name": "Peter Jackson", "job": "Director"}]',
            '[{"name": "George Lucas", "job": "Director"}]',
            '[{"name": "James Cameron", "job": "Director"}]',
            '[{"name": "James Cameron", "job": "Director"}]',
            '[{"name": "Joss Whedon", "job": "Director"}]',
            '[{"name": "Steven Spielberg", "job": "Director"}]',
            '[{"name": "Michael Curtiz", "job": "Director"}]',
            '[{"name": "Victor Fleming", "job": "Director"}]',
            '[{"name": "Victor Fleming", "job": "Director"}]',
            '[{"name": "Orson Welles", "job": "Director"}]',
            '[{"name": "Alfred Hitchcock", "job": "Director"}]'
        ],
        'vote_average': [  # Added IMDb-like ratings
            8.4, 8.3, 8.1, 8.5, 9.0,
            8.2, 9.2, 8.6, 8.5, 8.7,
            8.3, 7.6, 7.7, 7.8, 7.9,
            8.5, 8.0, 7.8, 8.3, 8.1
        ],
        'release_date': [  # Added release dates
            '2008-07-18', '2010-07-16', '1999-03-31', '1994-10-14', '1972-03-24',
            '1994-07-06', '1994-09-23', '1999-10-15', '1990-09-19', '2001-12-19',
            '1977-05-25', '1997-12-19', '2009-12-18', '2012-05-04', '1993-06-11',
            '1942-11-26', '1939-12-15', '1939-08-25', '1941-09-05', '1958-07-21'
        ]
    }
    
    return pd.DataFrame(sample_movies)

def safe_convert(obj):
    """Safely convert string representation of list to actual list"""
    if pd.isna(obj) or obj == '':
        return []
    
    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, list):
                return [item.get('name', '') for item in parsed if isinstance(item, dict) and 'name' in item]
        except (ValueError, SyntaxError):
            return []
    
    return []

def get_director(obj):
    """Extract director name from crew data"""
    if pd.isna(obj) or obj == '':
        return ''
    
    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, list):
                for person in parsed:
                    if isinstance(person, dict) and person.get('job') == 'Director':
                        return person.get('name', '')
        except (ValueError, SyntaxError):
            return ''
    
    return ''

def extract_year(release_date):
    """Extract year from release_date"""
    if pd.isna(release_date) or release_date == '':
        return 'Unknown'
    try:
        return str(pd.to_datetime(release_date).year)
    except:
        return 'Unknown'

@st.cache_data
def process_movie_features(movies_df):
    """Process movie features for recommendation"""
    movies_df = movies_df.copy()
    
    movies_df['genres'] = movies_df['genres'].apply(safe_convert)
    movies_df['keywords'] = movies_df['keywords'].apply(safe_convert)
    movies_df['cast'] = movies_df['cast'].apply(lambda x: safe_convert(x)[:3])  # top 3 actors
    movies_df['crew'] = movies_df['crew'].apply(get_director)
    
    # Extract release year
    movies_df['release_year'] = movies_df['release_date'].apply(extract_year)
    
    # Handle missing overview and vote_average
    movies_df['overview'] = movies_df['overview'].fillna('')
    movies_df['vote_average'] = movies_df['vote_average'].fillna(0.0)
    
    # Combine features into tags
    movies_df['tags'] = (
        movies_df['overview'] + ' ' +
        movies_df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
        movies_df['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
        movies_df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
        movies_df['crew']
    )
    
    # Convert to lowercase
    movies_df['tags'] = movies_df['tags'].apply(lambda x: x.lower())
    
    return movies_df

@st.cache_data
def compute_similarity_matrix(movies_df):
    """Compute similarity matrix"""
    cv = CountVectorizer(max_features=5000, stop_words='english')
    
    try:
        vectors = cv.fit_transform(movies_df['tags']).toarray()
        similarity = cosine_similarity(vectors)
        return similarity, True
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return None, False

# TMDB API configuration
API_KEY = "xyz"

@st.cache_data
def fetch_poster(movie_title):

    if not API_KEY or API_KEY == "API_KEY_HERE":
        return "https://via.placeholder.com/500x750?text=No+API+Key"
    
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        
        return "https://via.placeholder.com/500x750?text=No+Image"
    
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750?text=Connection+Error"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Error"

def recommend(movie_title, movies_df, similarity_matrix, genre_filter=None):
    """Generate movie recommendations"""
    if movie_title not in movies_df['title'].values:
        return [], [], [], []  # Added lists for ratings and years
    
    try:
        idx = movies_df[movies_df['title'] == movie_title].index[0]
        
        distances = list(enumerate(similarity_matrix[idx]))
        
        if genre_filter and genre_filter != "All":
            filtered_distances = []
            for i, score in distances:
                movie_genres = movies_df.iloc[i]['genres']
                if genre_filter in movie_genres:
                    filtered_distances.append((i, score))
            distances = filtered_distances
        
        movie_indices = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        
        recommended_titles = []
        recommended_posters = []
        recommended_ratings = []
        recommended_years = []
        recommended_directors = []
        
        for i, score in movie_indices:
            title = movies_df.iloc[i]['title']
            recommended_titles.append(title)
            recommended_posters.append(fetch_poster(title))
            recommended_ratings.append(movies_df.iloc[i]['vote_average'])
            recommended_years.append(movies_df.iloc[i]['release_year'])
            recommended_directors.append(movies_df.iloc[i]['crew'])
        
        return recommended_titles, recommended_posters, recommended_ratings, recommended_years, recommended_directors
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return [], [], [], [], []

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    with st.spinner("Loading movie data..."):
        movies_df, data_loaded = load_and_process_data()
        
        if not data_loaded:
            st.warning("Using sample data for demonstration. Please add your CSV files for full functionality.")
        
        movies_df = process_movie_features(movies_df)
        similarity_matrix, similarity_computed = compute_similarity_matrix(movies_df)
    
    if not similarity_computed:
        st.error("Could not compute similarity matrix. Please check your data.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        all_genres = set()
        for genre_list in movies_df['genres']:
            all_genres.update(genre_list)
        
        selected_genre = st.selectbox(
            "🎞️ Select Genre (Optional):",
            ["All"] + sorted(list(all_genres))
        )
    
    with col2:
        if selected_genre == "All":
            filtered_movies = movies_df['title'].values
        else:
            filtered_movies = movies_df[
                movies_df['genres'].apply(lambda x: selected_genre in x)
            ]['title'].values
        
        selected_movie = st.selectbox(
            "🎬 Select a Movie:",
            options=filtered_movies,
            help="Choose a movie to get recommendations"
        )
    
    # Display movie information
    if selected_movie:
        movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
        
        st.markdown("### Selected Movie Information")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            poster_url = fetch_poster(selected_movie)
            st.image(poster_url, width=200)
        
        with col2:
            st.markdown(f"**Title:** {movie_info['title']}")
            st.markdown(f"**Director:** {movie_info['crew']}")
            st.markdown(f"**Genres:** {', '.join(movie_info['genres'])}")
            st.markdown(f"**Cast:** {', '.join(movie_info['cast'][:3])}")
            st.markdown(f"**IMDb Rating:** {movie_info['vote_average']:.1f}/10")
            st.markdown(f"**Release Year:** {movie_info['release_year']}")
            if len(movie_info['overview']) > 0:
                st.markdown(f"**Overview:** {movie_info['overview'][:200]}...")
    
    if st.button('Get Recommendations', type="primary", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            names, posters, ratings, years, directors = recommend(selected_movie, movies_df, similarity_matrix, selected_genre)
        
        if names:
            st.markdown("### 🎬 Top 5 Recommendations:")
            
            cols = st.columns(5)
            for idx, (name, poster, rating, year, director) in enumerate(zip(names, posters, ratings, years, directors)):
                with cols[idx]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    st.image(poster, use_container_width=True)
                    st.markdown(f"**{name}**")
                    st.markdown(f"**Director:** {director}")
                    st.markdown(f"**Rating:** {rating:.1f}/10")
                    st.markdown(f"**Year:** {year}")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Please try a different movie or genre filter.")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions:
    1. **Select a genre** (optional) to filter movies by specific genre
    2. **Choose a movie** from the dropdown that you enjoyed
    3. **Click 'Get Recommendations'** to find similar movies
    4. **Enjoy** discovering new movies!
    
    ### 🔧 Setup Requirements:
    - Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the same directory
    - Get a free TMDB API key from [themoviedb.org](https://www.themoviedb.org/settings/api)
    - Replace the API_KEY variable with your actual key for poster images
    
    ### ⚡ Features:
    - Content-based filtering using movie metadata
    - Genre-based filtering
    - Movie poster integration
    - Fast similarity computation with caching
    - Displays IMDb rating, release year, and director for recommendations
    """)

if __name__ == "__main__":
    main()
