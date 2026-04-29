import pandas as pd
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PYTORCH CLASSES
# ==========================================
class MovieLensDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class SimplifiedNeuMF(nn.Module):
    def __init__(self, num_users, num_items, embed_size=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_size)
        self.item_embed = nn.Embedding(num_items, embed_size)
        
        self.mlp_pipeline = nn.Sequential(
            nn.Linear(embed_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.output = nn.Linear(embed_size + 32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        u = self.user_embed(user_indices)
        i = self.item_embed(item_indices)
        
        gmf_vector = u * i 
        mlp_vector = self.mlp_pipeline(torch.cat([u, i], dim=1))

        combined_vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        prediction = self.output(combined_vector)
        
        return (self.sigmoid(prediction) * 4.5 + 0.5).squeeze()

# ==========================================
# 2. THE MASTER TRAINING FUNCTION
# ==========================================
def train_hybrid_model(epochs=8, batch_size=4096):
    """
    Loads the processed parquet data, trains the NeuMF model, 
    and saves the trained weights to disk.
    """
    processed_dir = '../data/processed'
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading prepared master data...")
    df = pd.read_parquet(f'{processed_dir}/master_data_v2.parquet')
    
    # Sort chronologically to prevent data leakage
    df = df.sort_values('datetime')

    print("Encoding IDs...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['userId'])
    df['item_idx'] = item_encoder.fit_transform(df['movieId'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    # 80/20 Chronological Split
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]

    # Create Dataloader
    train_dataset = MovieLensDataset(train_df['user_idx'].values, train_df['item_idx'].values, train_df['rating'].values)
    # Using num_workers=4 and pin_memory=True drastically speeds up data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Users: {num_users} | Movies: {num_items}")

    model = SimplifiedNeuMF(num_users, num_items).to(device)
    criterion = nn.SmoothL1Loss() # Huber Loss
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    print(f"Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Huber): {avg_train_loss:.4f} | Time: {epoch_time:.2f}s")

    # SAVE THE MODEL
    model_path = f'{models_dir}/neumf_model.pth'
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train_hybrid_model()