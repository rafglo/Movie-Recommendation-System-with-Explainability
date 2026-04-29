# src/content_engine.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Global variables to cache the massive matrices in RAM
_dna_matrix = None
_content_df = None

def _initialize_engine():
    """Internal function to load and cache the Tag Genome only once."""
    global _dna_matrix, _content_df
    
    # If it's already loaded, skip this whole process
    if _dna_matrix is not None:
        return

    print("⚙️ Booting up Content Engine (Loading Tag Genome into RAM)...")
    raw_dir = '../data/raw'
    
    movies = pd.read_csv(f'{raw_dir}/movies.csv')
    genome_scores = pd.read_csv(f'{raw_dir}/genome-scores.csv')
    
    # Pivot to dense DNA matrix
    _dna_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance')
    
    # Filter movies to only those with DNA
    valid_movie_ids = _dna_matrix.index
    _content_df = movies[movies['movieId'].isin(valid_movie_ids)].copy()
    print("✅ Content Engine Ready!\n")

def get_content_recommendations(movie_title, top_n=5):
    """
    Returns top N similar movies based purely on Tag Genome DNA.
    Automatically handles initializing the engine if it hasn't been booted yet.
    """
    # Ensure the engine is booted
    _initialize_engine()
    
    # 1. Find the exact movieId
    idx_search = _content_df[_content_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if idx_search.empty:
        return f"Error: Movie '{movie_title}' not found in the Genome database."
    
    target_movie_id = idx_search.iloc[0]['movieId']
    exact_title = idx_search.iloc[0]['title']
    
    # 2. Extract DNA vector
    target_dna = _dna_matrix.loc[target_movie_id].values.reshape(1, -1)
    
    # 3. Calculate cosine similarity
    sim_scores = cosine_similarity(target_dna, _dna_matrix).flatten()
    
    # 4. Sort and get top N matches (skipping index 0 which is the movie itself)
    similar_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
    top_movie_ids = _dna_matrix.index[similar_indices]
    
    # 5. Format results
    results = _content_df[_content_df['movieId'].isin(top_movie_ids)].copy()
    
    score_mapping = pd.DataFrame({
        'movieId': top_movie_ids,
        'similarity_score': sim_scores[similar_indices]
    })
    
    results = results.merge(score_mapping, on='movieId')
    results = results.sort_values('similarity_score', ascending=False)
    
    # Round the score for cleaner display
    results['similarity_score'] = results['similarity_score'].round(3)
    
    return results[['title', 'genres', 'similarity_score']]

if __name__ == "__main__":
    # Quick test if the script is run directly
    print(get_content_recommendations("Matrix, The", top_n=5))