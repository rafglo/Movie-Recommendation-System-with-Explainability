import pandas as pd # data manipulation
import os # operating system interfaces

def prep_master_data():
    """
    Loads raw MovieLens data, processes genres into binary features, 
    and prepares a clean dataset for explainable models.
    """
    # Dynamic path resolution to maintain project structure consistency
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    # Ensuring the processed directory exists before saving
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading rating and movie datasets...")

    # Extraction of core interaction data and item metadata
    ratings = pd.read_csv(os.path.join(raw_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))
    
    # Conversion of Unix timestamps to readable datetime objects for temporal analysis
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')

    print("Extracting explicit genre features for SHAP explainability...")
    # 1. Multi-hot encode the genres (converts "Action|Sci-Fi" into a binary matrix)
    genre_dummies = movies['genres'].str.get_dummies(sep='|')
    
    # 2. Attach these new binary columns back to the movies dataframe
    movies_with_genres = pd.concat([movies, genre_dummies], axis=1)
    
    # We want to merge the movieId, title, AND all our new genre columns
    cols_to_merge = ['movieId', 'title'] + list(genre_dummies.columns)

    # Merging movie titles and binary genres into the interaction log
    master_df = ratings.merge(
        movies_with_genres[cols_to_merge],
        on='movieId',
        how='left'
    )

    # Sorting by user and time to facilitate sequential splitting
    master_df = master_df.sort_values(['userId', 'datetime'])

    # Serialization to Parquet format
    output_path = os.path.join(processed_dir, 'master_data_small.parquet')
    master_df.to_parquet(output_path)

    print(f"✅ Saved dataset to {output_path}")
    print(f"Shape: {master_df.shape}")
    print(f"Extracted {len(genre_dummies.columns)} unique genres.")

if __name__ == "__main__":
    prep_master_data()