# src/content_engine.py
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_tfidf_matrix = None
_content_df = None

def _initialize_engine():
    """Loads tags/genres and builds the TF-IDF matrix only once."""
    global _tfidf_matrix, _content_df
    
    if _tfidf_matrix is not None:
        return

    print("⚙️ Booting up Content Engine (Building TF-IDF Vectors)...")
    
    # --- THE PATH FIX ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_dir = os.path.join(project_root, 'data', 'raw')
    # --------------------
    
    movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))
    tags = pd.read_csv(os.path.join(raw_dir, 'tags.csv'))
    
    # Clean and aggregate tags
    # Clean and aggregate tags
    tags = tags.dropna(subset=['tag'])
    tags['tag'] = tags['tag'].str.lower().str.strip()
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Merge and build the 'metadata' corpus
    _content_df = movies.merge(movie_tags, on='movieId', how='left')
    _content_df['tag'] = _content_df['tag'].fillna('')
    _content_df['genres'] = _content_df['genres'].str.replace('|', ' ', regex=False).str.lower()
    
    # Weight genres heavier by adding them twice
    _content_df['metadata'] = _content_df['genres'] + " " + _content_df['genres'] + " " + _content_df['tag']
    
    # Build the sparse math matrix
    tfidf = TfidfVectorizer(stop_words='english')
    _tfidf_matrix = tfidf.fit_transform(_content_df['metadata'])
    
    print("✅ TF-IDF Content Engine Ready!\n")

def get_content_recommendations(movie_title, top_n=5):
    """Returns top N similar movies based purely on TF-IDF metadata."""
    _initialize_engine()
    
    idx_search = _content_df[_content_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if idx_search.empty:
        return f"Error: Movie '{movie_title}' not found in the database."
    
    # Get the index of the exact match within the dataframe
    idx = idx_search.index[0]
    
    # Calculate cosine similarity against the whole matrix
    sim_scores = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    
    # Sort and get top N matches (skipping index 0 which is itself)
    similar_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
    
    results = _content_df.iloc[similar_indices][['title', 'genres']].copy()
    results['similarity_score'] = sim_scores[similar_indices].round(3)
    
    return results

if __name__ == "__main__":
    print(get_content_recommendations("Matrix, The", top_n=5))