# src/neural_cf.py

import pandas as pd # data manipulation
import os # operating system interfaces
import time # time-related functions
import pickle # object serialization
import numpy as np # numerical computing

import torch # tensor library and deep learning
import torch.nn as nn # neural network modules
import torch.optim as optim # optimization algorithms
from torch.utils.data import Dataset, DataLoader # data handling
from sklearn.preprocessing import LabelEncoder # encoding categorical labels
from sklearn.metrics import mean_squared_error, mean_absolute_error # model evaluation metrics


class MovieLensDataset(Dataset):
    """
    Encapsulates MovieLens data for PyTorch DataLoaders.
    Converts user/item indices and ratings into tensors.
    """
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class SimplifiedNeuMF(nn.Module):
    """
    Simplified Neural Matrix Factorization (NeuMF).
    Combines GMF (linear dot product) and MLP (non-linear deep features) 
    to capture diverse interaction patterns.
    """
    def __init__(self, num_users, num_items, embed_size=64):
        super().__init__()

        # Embedding layers with +1 buffer to accommodate the Unknown (UNK) token
        self.user_embed = nn.Embedding(num_users + 1, embed_size)
        self.item_embed = nn.Embedding(num_items + 1, embed_size)

        # MLP Pipeline: Processes concatenated user-item embeddings
        self.mlp_pipeline = nn.Sequential(
            nn.Linear(embed_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout used to mitigate overfitting on sparse data
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Final layer connecting GMF and MLP outputs
        self.output = nn.Linear(embed_size + 32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        u = self.user_embed(user_indices)
        i = self.item_embed(item_indices)

        # GMF Pathway: Element-wise product representing hidden factor overlap
        gmf_vector = u * i
        
        # MLP Pathway: Deep non-linear transformation of concatenated features
        mlp_vector = self.mlp_pipeline(torch.cat([u, i], dim=1))

        # Concatenation of linear and non-linear components
        combined_vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        prediction = self.output(combined_vector)

        # Scaling: Transforms 0-1 sigmoid range to 0.5-5.0 rating scale
        return (self.sigmoid(prediction) * 4.5 + 0.5).squeeze()


def safe_transform(encoder, values):
    """
    Label encoding that maps unseen values in the test set to an 'UNK' (0) index.
    This prevents crashes during inference on new users or movies.
    """
    mapping = {k: i + 1 for i, k in enumerate(encoder.classes_)}
    return np.array([mapping.get(v, 0) for v in values])


def prepare_split(df):
    """
    Temporal per-user split. 
    Allocates the first 80% of a user's history to training and the final 20% to testing.
    """
    df = df.sort_values(['userId', 'datetime'])
    df['rank'] = df.groupby('userId').cumcount()
    df['count'] = df.groupby('userId')['userId'].transform('count')

    train_df = df[df['rank'] < df['count'] * 0.8].copy()
    test_df = df[df['rank'] >= df['count'] * 0.8].copy()

    return train_df, test_df


def train_hybrid_model(epochs=10, batch_size=2048):
    """
    Orchestrates the model training process.
    Handles directory setup, data splitting, encoder persistence, and the training loop.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    parquet_path = os.path.join(processed_dir, 'master_data_small.parquet')
    global_df = pd.read_parquet(parquet_path)

    # Separation of data into training and validation sets before any encoding
    train_df, test_df = prepare_split(global_df)

    # Fitting encoders only on training data to prevent label leakage
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_encoder.fit(train_df['userId'])
    item_encoder.fit(train_df['movieId'])

    # Serializing encoders for consistent mapping during later evaluation
    with open(os.path.join(models_dir, 'user_encoder.pkl'), 'wb') as f:
        pickle.dump(user_encoder, f)

    with open(os.path.join(models_dir, 'item_encoder.pkl'), 'wb') as f:
        pickle.dump(item_encoder, f)

    # Transforming IDs to tensors using the UNK-safe mapping
    train_users = safe_transform(user_encoder, train_df['userId'])
    train_items = safe_transform(item_encoder, train_df['movieId'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    # Preparation of the data pipeline for high-volume training
    train_dataset = MovieLensDataset(train_users, train_items, train_df['rating'].values)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Users: {num_users} | Movies: {num_items}")

    model = SimplifiedNeuMF(num_users, num_items).to(device)

    # Using SmoothL1Loss for robustness against outliers
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (users, items, ratings) in enumerate(train_loader):
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)

            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), os.path.join(models_dir, 'neumf_model_small.pth'))
    print("Model saved!")


def evaluate_cf_model(batch_size=2048):
    """
    Performs evaluation on the test set and calculates standard regression metrics.
    """
    print("Loading test data and trained model...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')

    parquet_path = os.path.join(processed_dir, 'master_data_small.parquet')
    global_df = pd.read_parquet(parquet_path)

    # Loading the exact encoders used during training to ensure ID consistency
    with open(os.path.join(models_dir, 'user_encoder.pkl'), 'rb') as f:
        user_encoder = pickle.load(f)

    with open(os.path.join(models_dir, 'item_encoder.pkl'), 'rb') as f:
        item_encoder = pickle.load(f)

    # Re-generating the temporal split to access test interactions
    train_df, test_df = prepare_split(global_df)

    # Applying the safe transformation to handle any new movies/users in the test set
    test_users = safe_transform(user_encoder, test_df['userId'])
    test_items = safe_transform(item_encoder, test_df['movieId'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    test_dataset = MovieLensDataset(test_users, test_items, test_df['rating'].values)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplifiedNeuMF(num_users, num_items).to(device)

    # Loading the trained weights with strict state dictionary verification
    model_path = os.path.join(models_dir, 'neumf_model_small.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    actuals = []
    predictions = []

    print("Running predictions...")

    with torch.no_grad(): # Disabling gradient tracking to save memory and compute
        for users, items, ratings in test_loader:
            users, items = users.to(device), items.to(device)
            preds = model(users, items).cpu().numpy()

            predictions.extend(preds)
            actuals.extend(ratings.numpy())

    # Metric calculation for rating accuracy assessment
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print("\n========================================")
    print("EVALUATION RESULTS")
    print("========================================")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print("========================================\n")