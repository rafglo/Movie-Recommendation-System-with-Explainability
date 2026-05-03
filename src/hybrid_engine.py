# src/hybrid_engine.py

import os
import pickle
import torch
import pandas as pd
import numpy as np

# Import the components we built
from src.neural_cf import ExplainableNeuMF, safe_transform
from src.content_engine import get_content_recommendations

class HybridRecommender:
    def __init__(self):
        print("⚙️ Booting up Hybrid Recommender Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Setup Paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.models_dir = os.path.join(project_root, 'models')
        processed_dir = os.path.join(project_root, 'data', 'processed')
        
        # 2. Load the Master Data (We need this to lookup genres for the Deep Learning model)
        parquet_path = os.path.join(processed_dir, 'master_data_small.parquet')
        self.master_df = pd.read_parquet(parquet_path)
        
        # Drop duplicates so we have a clean lookup table of movieId -> genres
        self.movie_lookup = self.master_df.drop_duplicates(subset=['movieId']).set_index('movieId')
        
        # 3. Load Encoders
        with open(os.path.join(self.models_dir, 'user_encoder.pkl'), 'rb') as f:
            self.user_encoder = pickle.load(f)
        with open(os.path.join(self.models_dir, 'item_encoder.pkl'), 'rb') as f:
            self.item_encoder = pickle.load(f)
        with open(os.path.join(self.models_dir, 'genre_cols.pkl'), 'rb') as f:
            self.genre_cols = pickle.load(f)
            
        # 4. Boot the Neural Brain
        num_users = len(self.user_encoder.classes_)
        num_items = len(self.item_encoder.classes_)
        num_genres = len(self.genre_cols)
        
        self.model = ExplainableNeuMF(num_users, num_items, num_genres).to(self.device)
        model_path = os.path.join(self.models_dir, 'neumf_model_small.pth')
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.eval()
        
        print("✅ Hybrid Engine Ready!")

    def recommend(self, user_id, movie_title, top_n=5, candidate_pool_size=50):
        """
        The Master Inference Function.
        1. Uses TF-IDF to find 50 similar movies.
        2. Uses PyTorch to predict how much the specific user will like them.
        3. Sorts and returns the Top N.
        """
        # --- STEP 1: CANDIDATE GENERATION (Content Engine) ---
        # TF-IDF acts as our fast, lightweight filter
        candidates_df = get_content_recommendations(movie_title, top_n=candidate_pool_size)
        
        if isinstance(candidates_df, str):  # Catches the "Movie not found" error string
            return candidates_df
        if candidates_df is None or candidates_df.empty:
            return "Movie not found in database."

        candidate_movie_ids = candidates_df['movieId'].values
        
        # --- STEP 2: FEATURE PREPARATION ---
        # Encode User (Repeat the user ID for every candidate movie)
        user_array = np.array([user_id] * len(candidate_movie_ids))
        user_tensor = torch.tensor(safe_transform(self.user_encoder, user_array), dtype=torch.long).to(self.device)
        
        # Encode Items
        item_tensor = torch.tensor(safe_transform(self.item_encoder, candidate_movie_ids), dtype=torch.long).to(self.device)
        
        # Extract Genres for the Wide Path
        # Look up the 1s and 0s matrix for our specific candidate movies
        candidate_genres = self.movie_lookup.loc[candidate_movie_ids, self.genre_cols].values
        genre_tensor = torch.tensor(candidate_genres, dtype=torch.float32).to(self.device)

        # --- STEP 3: NEURAL RANKING ---
        with torch.no_grad():
            # Pass all three features into the Explainable NeuMF model
            predictions = self.model(user_tensor, item_tensor, genre_tensor).cpu().numpy()
            
        # --- STEP 4: FORMAT OUTPUT ---
        candidates_df['predicted_rating'] = predictions
        
        # Sort by the Neural Network's prediction, not the original TF-IDF similarity!
        final_recs = candidates_df.sort_values(by='predicted_rating', ascending=False).head(top_n)
        
        return final_recs[['title', 'genres', 'similarity_score', 'predicted_rating']]

if __name__ == "__main__":
    # A quick test you can run directly to verify the bridge is working
    engine = HybridRecommender()
    
    # Test how User 1 feels about The Matrix compared to User 50
    print(f"\n🎬 Top Picks for User #1 based on 'The Matrix':")
    print(engine.recommend(user_id=1, movie_title="Matrix, The"))
    
    print(f"\n🎬 Top Picks for User #50 based on 'The Matrix':")
    print(engine.recommend(user_id=50, movie_title="Matrix, The"))