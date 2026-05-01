import pandas as pd
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

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
    def __init__(self, num_users, num_items, embed_size=32): # Smaller embeddings for small data
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_size)
        self.item_embed = nn.Embedding(num_items, embed_size)
        
        self.mlp_pipeline = nn.Sequential(
            nn.Linear(embed_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
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

def train_hybrid_model(epochs=5, batch_size=256):
    processed_dir = '../data/processed'
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    df = pd.read_parquet(f'{processed_dir}/master_data_small.parquet')
    df = df.sort_values('datetime')

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['userId'])
    df['item_idx'] = item_encoder.fit_transform(df['movieId'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]

    train_dataset = MovieLensDataset(train_df['user_idx'].values, train_df['item_idx'].values, train_df['rating'].values)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Users: {num_users} | Movies: {num_items}")

    model = SimplifiedNeuMF(num_users, num_items).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

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
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), f'{models_dir}/neumf_model_small.pth')
    print("✅ Model saved!")

if __name__ == "__main__":
    train_hybrid_model()