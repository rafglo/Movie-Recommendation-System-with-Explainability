import pandas as pd
import os

def prep_master_data(min_ratings=10):
    """Loads the 100k dataset and cleans it for the Gold Standard split."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading 100k rating dataset...")
    ratings = pd.read_csv(os.path.join(raw_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))

    # Filter cold start
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()

    active_users = user_counts[user_counts >= min_ratings].index
    warm_movies = movie_counts[movie_counts >= min_ratings].index

    ratings = ratings[ratings['userId'].isin(active_users)]
    ratings = ratings[ratings['movieId'].isin(warm_movies)]

    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    master_df = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')

    output_path = os.path.join(processed_dir, 'master_data_small.parquet')
    master_df.to_parquet(output_path)
    print(f"✅ Saved 100k dataset to {output_path}")

if __name__ == "__main__":
    prep_master_data()