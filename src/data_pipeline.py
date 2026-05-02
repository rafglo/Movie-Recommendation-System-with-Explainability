import pandas as pd # data manipulation
import os # operating system interfaces

def prep_master_data():
    """
    Loads raw MovieLens data and prepares a clean dataset.
    """
    # Dynamic path resolution to maintain project structure consistency
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    # Ensuring the processed directory exists before saving
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading rating dataset")

    # Extraction of core interaction data and item metadata
    ratings = pd.read_csv(os.path.join(raw_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))
    
    # Conversion of Unix timestamps to readable datetime objects for temporal analysis
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # Merging movie titles into the interaction log
    master_df = ratings.merge(
        movies[['movieId', 'title']],
        on='movieId',
        how='left'
    )

    # Sorting by user and time to facilitate sequential splitting
    master_df = master_df.sort_values(['userId', 'datetime'])

    # Serialization to Parquet format
    output_path = os.path.join(processed_dir, 'master_data_small.parquet')
    master_df.to_parquet(output_path)

    print(f"Saved dataset to {output_path}")
    print(f"Shape: {master_df.shape}")
    print("NOTE: Filtering must be applied AFTER train/test split.")

if __name__ == "__main__":
    prep_master_data()