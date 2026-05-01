import pandas as pd
import os

def prep_master_data(target_rows=1000000, min_ratings=20):
    """
    Loads the massive 33M dataset and strategically downsamples it to ~1M rows
    by sampling USERS, not rows. This preserves the dense matrix required for Deep Learning.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading massive 33M rating dataset (This might take a minute)...")
    ratings = pd.read_csv(os.path.join(raw_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))

    print(f"Original shape: {len(ratings):,} rows.")

    # 1. Filter out absolute cold-start users/movies first
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()

    active_users = user_counts[user_counts >= min_ratings].index
    warm_movies = movie_counts[movie_counts >= min_ratings].index

    ratings = ratings[ratings['userId'].isin(active_users)]
    ratings = ratings[ratings['movieId'].isin(warm_movies)]

    # 2. The Smart Downsample: Target ~1M rows by sampling Users
    current_rows = len(ratings)
    if current_rows > target_rows:
        # Calculate the exact fraction of users we need to hit our 1M row target
        keep_fraction = target_rows / current_rows
        
        # Randomly sample that fraction of unique users
        unique_users = pd.Series(ratings['userId'].unique())
        sampled_users = unique_users.sample(frac=keep_fraction, random_state=42)
        
        # Keep ONLY the ratings belonging to those chosen users
        ratings = ratings[ratings['userId'].isin(sampled_users)]

    print(f"Downsampled shape: {len(ratings):,} rows. (Preserved dense user histories!)")

    # 3. Create datetime and merge titles
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    master_df = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')

    # 4. Save to processed folder
    output_path = os.path.join(processed_dir, 'master_data_1M.parquet')
    master_df.to_parquet(output_path)
    print(f"✅ Saved 1M dataset to {output_path}")

if __name__ == "__main__":
    prep_master_data()