import pandas as pd
import os

def prep_master_data(min_user_ratings=500, min_movie_ratings=500):
    """
    Loads raw 33M data, filters down to power users and blockbusters, 
    and saves a lightweight parquet file for training.
    """
    raw_dir = '../data/raw'
    processed_dir = '../data/processed'
 
    os.makedirs(processed_dir, exist_ok=True)

    ratings = pd.read_csv(f'{raw_dir}/ratings.csv')
    movies = pd.read_csv(f'{raw_dir}/movies.csv')

    # 1. Filter for Super Users
    user_counts = ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings['userId'].isin(active_users)]

    # 2. Filter for Blockbuster Movies
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings = ratings[ratings['movieId'].isin(popular_movies)]

    # 3. Create datetime and merge titles
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    master_df = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')

    # 4. Save to processed folder
    output_path = f'{processed_dir}/master_data_v2.parquet'
    master_df.to_parquet(output_path)

if __name__ == "__main__":
    prep_master_data()