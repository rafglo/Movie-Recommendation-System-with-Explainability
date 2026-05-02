# src/content_engine.py

import pandas as pd # data manipulation
import os # operating system interfaces
from sklearn.feature_extraction.text import TfidfVectorizer # text vectorization using TF-IDF
from sklearn.metrics.pairwise import cosine_similarity # similarity measurement between vectors

_tfidf_matrix = None
_content_df = None

_tfidf_matrix = None
_content_df = None

def _initialize_engine():
    """
    Loads data and builds the TF-IDF matrix. Uses global variables to act as 
    a singleton, preventing redundant matrix calculations across calls.
    """
    global _tfidf_matrix, _content_df
    
    if _tfidf_matrix is not None:
        return
    # Resolution to ensure the script finds the data directory regardless
    # of where the execution was triggered in the project tree
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_dir = os.path.join(project_root, 'data', 'raw')
    
    movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))
    tags = pd.read_csv(os.path.join(raw_dir, 'tags.csv'))
    
    # Cleaning and normalization of user-generated tags
    tags = tags.dropna(subset=['tag'])
    tags['tag'] = tags['tag'].str.lower().str.strip()
    
    # Aggregating multiple tags per movie into a single string
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Merging datasets and preparing the final metadata corpus
    _content_df = movies.merge(movie_tags, on='movieId', how='left')
    _content_df['tag'] = _content_df['tag'].fillna('')
    _content_df['genres'] = _content_df['genres'].str.replace('|', ' ', regex=False).str.lower()
    
    # Feature weighting: Genres are duplicated to increase their significance 
    # in the TF-IDF vector relative to individual user tags
    _content_df['metadata'] = _content_df['genres'] + " " + _content_df['genres'] + " " + _content_df['tag']
    
    # Transforming the text corpus into a mathematical sparse matrix
    tfidf = TfidfVectorizer(stop_words='english')
    _tfidf_matrix = tfidf.fit_transform(_content_df['metadata'])
    
def get_content_recommendations(movie_title, top_n=5):
    """
    Retrieves the top N movies most similar to the input title using 
    cosine similarity on the pre-computed TF-IDF matrix.
    """
    _initialize_engine()
    
    # Searching for the movie index using case-insensitive matching
    idx_search = _content_df[_content_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if idx_search.empty:
        return f"Error: Movie '{movie_title}' not found in the database."
    
    # Selecting the first match found in the dataframe
    idx = idx_search.index[0]
    
    # Computing Cosine Similarity between the target vector and the entire matrix
    sim_scores = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    
    # Sorting indices based on similarity scores, excluding the input movie itself
    similar_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
    
    # Returning slice of dataframe with calculated similarity metrics
    results = _content_df.iloc[similar_indices][['title', 'genres']].copy()
    results['similarity_score'] = sim_scores[similar_indices].round(3)
    
    return results

if __name__ == "__main__":
    print(get_content_recommendations("Matrix, The", top_n=5))