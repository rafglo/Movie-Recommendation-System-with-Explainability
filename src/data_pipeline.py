import pandas as pd
import os

def prep_master_data(min_movie_ratings=10):
    """
    Loads raw 100k data, applies a light filter to remove pure cold-start items 
    from the CF training set, and saves a parquet file.
    """
    raw_dir = '../data/raw'
    processed_dir = '../data/processed'
 
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading 100k rating dataset...")
    ratings = pd.read_csv(f'{raw_dir}/ratings.csv')
    movies = pd.read_csv(f'{raw_dir}/movies.csv')

    # 1. Light filter: Keep movies with at least a few ratings
    movie_counts = ratings['movieId'].value_counts()
    warm_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings = ratings[ratings['movieId'].isin(warm_movies)]

    # 2. Create datetime and merge titles
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    master_df = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')

    # 3. Save to processed folder
    output_path = f'{processed_dir}/master_data_small.parquet'
    master_df.to_parquet(output_path)
    print(f"✅ Saved processed data to {output_path}")

if __name__ == "__main__":
    prep_master_data()