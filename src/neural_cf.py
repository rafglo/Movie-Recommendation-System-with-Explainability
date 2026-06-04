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
    Converts user indices, item indices, explicit genres, and ratings into tensors.
    """
    def __init__(self, users, items, genres, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.genres = torch.tensor(genres, dtype=torch.float32) # Added genre tensor
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        # Now returns 4 elements per batch
        return self.users[idx], self.items[idx], self.genres[idx], self.ratings[idx]


class ExplainableNeuMF(nn.Module):
    """
    Context-Aware Neural Matrix Factorization.
    """
    def __init__(self, num_users, num_items, num_genres, embed_size=64):
        super().__init__()

        # Embedding layers with +1 buffer to accommodate the Unknown (UNK) token
        self.user_embed = nn.Embedding(num_users + 1, embed_size)
        self.item_embed = nn.Embedding(num_items + 1, embed_size)

        # MLP Pipeline: Processes concatenated user-item embeddings
        self.mlp_pipeline = nn.Sequential(
            nn.Linear(embed_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Final layer dynamically accepts the GMF, MLP, AND the explicit genre features
        self.output = nn.Linear(embed_size + 32 + num_genres, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices, explicit_genres):
        u = self.user_embed(user_indices)
        i = self.item_embed(item_indices)

        # GMF Pathway
        gmf_vector = u * i
        
        # MLP Pathway
        mlp_vector = self.mlp_pipeline(torch.cat([u, i], dim=1))

        # WIDE & DEEP CONCATENATION: Deep features + Human-readable explicit genres
        combined_vector = torch.cat([gmf_vector, mlp_vector, explicit_genres], dim=1)
        prediction = self.output(combined_vector)

        # Scaling: Transforms 0-1 sigmoid range to 0.5-5.0 rating scale
        return (self.sigmoid(prediction) * 4.5 + 0.5).squeeze()


class RankingNeuMF(nn.Module):
    """
    NeuMF variant for top-K ranking. It returns raw relevance logits instead
    of scaled star ratings, so it can be trained with BCEWithLogitsLoss.
    """
    def __init__(self, num_users, num_items, num_genres, embed_size=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users + 1, embed_size)
        self.item_embed = nn.Embedding(num_items + 1, embed_size)

        self.mlp_pipeline = nn.Sequential(
            nn.Linear(embed_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.output = nn.Linear(embed_size + 32 + num_genres, 1)

    def forward(self, user_indices, item_indices, explicit_genres):
        u = self.user_embed(user_indices)
        i = self.item_embed(item_indices)
        gmf_vector = u * i
        mlp_vector = self.mlp_pipeline(torch.cat([u, i], dim=1))
        combined_vector = torch.cat([gmf_vector, mlp_vector, explicit_genres], dim=1)
        return self.output(combined_vector).squeeze()


class RankingMovieLensDataset(Dataset):
    """
    Positive and sampled-negative interaction dataset for ranking NeuMF.
    Labels are binary relevance targets rather than star ratings.
    """
    def __init__(self, users, items, genres, labels):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.genres = torch.tensor(genres, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.genres[idx], self.labels[idx]


def safe_transform(encoder, values):
    """
    Label encoding that maps unseen values in the test set to an 'UNK' (0) index.
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


def extract_genre_features(df):
    """
    Helper function to dynamically identify the new genre columns
    by filtering out the standard metadata columns.
    """
    standard_cols = ['userId', 'movieId', 'rating', 'timestamp', 'datetime', 'title', 'rank', 'count']
    genre_cols = [col for col in df.columns if col not in standard_cols]
    return genre_cols


def load_master_data_for_training():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    processed_path = os.path.join(project_root, 'data', 'processed', 'master_data_small.parquet')

    try:
        df = pd.read_parquet(processed_path)
    except (ImportError, FileNotFoundError):
        raw_dir = os.path.join(project_root, 'data', 'raw')
        ratings = pd.read_csv(os.path.join(raw_dir, 'ratings.csv'))
        movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        genre_dummies = movies['genres'].str.get_dummies(sep='|')
        movies_with_genres = pd.concat([movies, genre_dummies], axis=1)
        cols_to_merge = ['movieId', 'title'] + list(genre_dummies.columns)
        df = ratings.merge(
            movies_with_genres[cols_to_merge],
            on='movieId',
            how='left'
        ).sort_values(['userId', 'datetime'])

    genre_cols = extract_genre_features(df)
    if genre_cols:
        return df

    raw_dir = os.path.join(project_root, 'data', 'raw')
    movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))
    genre_dummies = movies['genres'].str.get_dummies(sep='|')
    movies_with_genres = pd.concat([movies[['movieId']], genre_dummies], axis=1)
    repaired_df = df.drop(columns=list(genre_dummies.columns), errors='ignore').merge(
        movies_with_genres,
        on='movieId',
        how='left'
    )
    repaired_df[list(genre_dummies.columns)] = repaired_df[list(genre_dummies.columns)].fillna(0)
    return repaired_df.sort_values(['userId', 'datetime'])


def build_ranking_training_frame(train_df, all_movie_ids, min_positive_rating=4.0, negatives_per_positive=4, random_state=42):
    rng = np.random.default_rng(random_state)
    train_seen = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    positive_df = train_df[train_df['rating'] >= min_positive_rating][['userId', 'movieId']].copy()
    positive_df['label'] = 1.0

    negative_rows = []
    all_movie_ids = np.array(sorted(all_movie_ids))

    for user_id, user_positive_df in positive_df.groupby('userId'):
        seen = train_seen.get(user_id, set())
        negative_pool = np.array([movie_id for movie_id in all_movie_ids if movie_id not in seen])
        if len(negative_pool) == 0:
            continue

        sample_size = len(user_positive_df) * negatives_per_positive
        sampled = rng.choice(
            negative_pool,
            size=sample_size,
            replace=len(negative_pool) < sample_size
        )
        negative_rows.extend((user_id, int(movie_id), 0.0) for movie_id in sampled)

    negative_df = pd.DataFrame(negative_rows, columns=['userId', 'movieId', 'label'])
    return pd.concat([positive_df, negative_df], ignore_index=True)


def evaluate_ranking_model_topk(
    model,
    df,
    train_df,
    test_df,
    user_encoder,
    item_encoder,
    genre_cols,
    device,
    k=10,
    min_positive_rating=4.0,
    negatives_per_user=100,
    max_users=100,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    all_items = set(df['movieId'].unique())
    movie_lookup = df.drop_duplicates(subset=['movieId']).set_index('movieId')
    train_seen = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    positives_by_user = (
        test_df[test_df['rating'] >= min_positive_rating]
        .groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )
    users = sorted(positives_by_user.keys())
    if max_users is not None and len(users) > max_users:
        users = sorted(rng.choice(users, size=max_users, replace=False).tolist())

    ndcg_values = []
    precision_values = []
    recall_values = []
    hit_values = []
    model.eval()

    with torch.no_grad():
        for user_id in users:
            positives = positives_by_user[user_id]
            seen = train_seen.get(user_id, set())
            negative_pool = np.array(list(all_items - seen - positives))
            if negatives_per_user is None or len(negative_pool) <= negatives_per_user:
                negatives = set(negative_pool.tolist())
            else:
                negatives = set(rng.choice(negative_pool, size=negatives_per_user, replace=False).tolist())

            candidates = list(positives | negatives)
            if not candidates:
                continue

            user_array = np.array([user_id] * len(candidates))
            item_array = np.array(candidates)
            user_tensor = torch.tensor(safe_transform(user_encoder, user_array), dtype=torch.long).to(device)
            item_tensor = torch.tensor(safe_transform(item_encoder, item_array), dtype=torch.long).to(device)
            genre_tensor = torch.tensor(
                movie_lookup.loc[item_array, genre_cols].values,
                dtype=torch.float32
            ).to(device)
            scores = model(user_tensor, item_tensor, genre_tensor).cpu().numpy()
            ranked_items = [item for item, _ in sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)]
            top_k = ranked_items[:k]
            hits = [1 if item in positives else 0 for item in top_k]
            hit_count = sum(hits)
            ideal_hits = min(len(positives), k)
            discounts = np.log2(np.arange(2, len(hits) + 2))
            dcg = float(np.sum(np.array(hits) / discounts)) if hits else 0.0
            idcg = float(np.sum(np.ones(ideal_hits) / np.log2(np.arange(2, ideal_hits + 2)))) if ideal_hits else 0.0

            precision_values.append(hit_count / k)
            recall_values.append(hit_count / len(positives))
            hit_values.append(1.0 if hit_count > 0 else 0.0)
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f'precision@{k}': float(np.mean(precision_values)) if precision_values else 0.0,
        f'recall@{k}': float(np.mean(recall_values)) if recall_values else 0.0,
        f'hit_rate@{k}': float(np.mean(hit_values)) if hit_values else 0.0,
        f'ndcg@{k}': float(np.mean(ndcg_values)) if ndcg_values else 0.0,
    }


def train_neumf_ranking_model(
    epochs=5,
    batch_size=1024,
    negatives_per_positive=4,
    min_positive_rating=4.0,
    learning_rate=0.001,
    validation_k=10,
    validation_users=100,
    validation_negatives=100,
    random_state=42,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    global_df = load_master_data_for_training()
    genre_cols = extract_genre_features(global_df)
    train_df, test_df = prepare_split(global_df)
    all_movie_ids = global_df['movieId'].unique()

    ranking_df = build_ranking_training_frame(
        train_df,
        all_movie_ids=all_movie_ids,
        min_positive_rating=min_positive_rating,
        negatives_per_positive=negatives_per_positive,
        random_state=random_state,
    )
    item_genres = global_df.drop_duplicates(subset=['movieId']).set_index('movieId')[genre_cols]
    ranking_df = ranking_df.merge(
        item_genres.reset_index(),
        on='movieId',
        how='left',
    )
    ranking_df[genre_cols] = ranking_df[genre_cols].fillna(0)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    user_encoder.fit(train_df['userId'])
    item_encoder.fit(train_df['movieId'])

    with open(os.path.join(models_dir, 'ranking_user_encoder.pkl'), 'wb') as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(models_dir, 'ranking_item_encoder.pkl'), 'wb') as f:
        pickle.dump(item_encoder, f)
    with open(os.path.join(models_dir, 'ranking_genre_cols.pkl'), 'wb') as f:
        pickle.dump(genre_cols, f)

    users = safe_transform(user_encoder, ranking_df['userId'])
    items = safe_transform(item_encoder, ranking_df['movieId'])
    genres = ranking_df[genre_cols].values
    labels = ranking_df['label'].values

    dataset = RankingMovieLensDataset(users, items, genres, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RankingNeuMF(
        num_users=len(user_encoder.classes_),
        num_items=len(item_encoder.classes_),
        num_genres=len(genre_cols),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_ndcg = -1.0
    best_path = os.path.join(models_dir, 'neumf_ranking_model.pth')
    history_rows = []

    print(
        f"Training ranking NeuMF on {device} | "
        f"Interactions: {len(ranking_df)} | Positives: {int(labels.sum())} | "
        f"Negatives: {int((labels == 0).sum())}"
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for users_batch, items_batch, genres_batch, labels_batch in loader:
            users_batch = users_batch.to(device)
            items_batch = items_batch.to(device)
            genres_batch = genres_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(users_batch, items_batch, genres_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        metrics = evaluate_ranking_model_topk(
            model=model,
            df=global_df,
            train_df=train_df,
            test_df=test_df,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            genre_cols=genre_cols,
            device=device,
            k=validation_k,
            min_positive_rating=min_positive_rating,
            negatives_per_user=validation_negatives,
            max_users=validation_users,
            random_state=random_state,
        )
        ndcg = metrics[f'ndcg@{validation_k}']

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), best_path)

        history_rows.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            f'precision@{validation_k}': metrics[f'precision@{validation_k}'],
            f'recall@{validation_k}': metrics[f'recall@{validation_k}'],
            f'hit_rate@{validation_k}': metrics[f'hit_rate@{validation_k}'],
            f'ndcg@{validation_k}': ndcg,
            'is_best': ndcg == best_ndcg,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
            f"NDCG@{validation_k}: {ndcg:.4f} | "
            f"Precision@{validation_k}: {metrics[f'precision@{validation_k}']:.4f} | "
            f"Time: {time.time() - start_time:.2f}s"
        )

    print(f"Saved best ranking NeuMF to {best_path} | Best NDCG@{validation_k}: {best_ndcg:.4f}")
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    history_path = os.path.join(reports_dir, 'neumf_ranking_training_history.csv')
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    print(f"Saved ranking training history to {history_path}")
    return best_ndcg


def train_hybrid_model(epochs=10, batch_size=2048):
    """
    Orchestrates the model training process for the Explainable network.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    parquet_path = os.path.join(processed_dir, 'master_data_small.parquet')
    global_df = pd.read_parquet(parquet_path)

    # Automatically detect the binary genre columns generated by the data pipeline
    genre_cols = extract_genre_features(global_df)
    num_genres = len(genre_cols)

    train_df, test_df = prepare_split(global_df)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_encoder.fit(train_df['userId'])
    item_encoder.fit(train_df['movieId'])

    with open(os.path.join(models_dir, 'user_encoder.pkl'), 'wb') as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(models_dir, 'item_encoder.pkl'), 'wb') as f:
        pickle.dump(item_encoder, f)
    
    # Save the specific genre column names so the Inference Engine knows them later
    with open(os.path.join(models_dir, 'genre_cols.pkl'), 'wb') as f:
        pickle.dump(genre_cols, f)

    train_users = safe_transform(user_encoder, train_df['userId'])
    train_items = safe_transform(item_encoder, train_df['movieId'])
    train_genres = train_df[genre_cols].values # Extract the 1s and 0s matrix

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    # Inject the genres into the Dataset
    train_dataset = MovieLensDataset(train_users, train_items, train_genres, train_df['rating'].values)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Users: {num_users} | Movies: {num_items} | Genres: {num_genres}")

    # Boot the new Explainable architecture
    model = ExplainableNeuMF(num_users, num_items, num_genres).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        # Unpack the 4 distinct elements from the new Dataloader
        for batch_idx, (users, items, genres, ratings) in enumerate(train_loader):
            users = users.to(device)
            items = items.to(device)
            genres = genres.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            
            # Pass the 3 features to the forward function
            predictions = model(users, items, genres)
            
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), os.path.join(models_dir, 'neumf_model_small.pth'))


def evaluate_cf_model(batch_size=2048):
    """
    Performs evaluation on the test set for the Explainable network.
    """
    print("Loading test data and trained model")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')

    parquet_path = os.path.join(processed_dir, 'master_data_small.parquet')
    global_df = pd.read_parquet(parquet_path)

    with open(os.path.join(models_dir, 'user_encoder.pkl'), 'rb') as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(models_dir, 'item_encoder.pkl'), 'rb') as f:
        item_encoder = pickle.load(f)
    
    # Load the genre column names to ensure strict alignment
    with open(os.path.join(models_dir, 'genre_cols.pkl'), 'rb') as f:
        genre_cols = pickle.load(f)

    num_genres = len(genre_cols)

    train_df, test_df = prepare_split(global_df)

    test_users = safe_transform(user_encoder, test_df['userId'])
    test_items = safe_transform(item_encoder, test_df['movieId'])
    test_genres = test_df[genre_cols].values

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    test_dataset = MovieLensDataset(test_users, test_items, test_genres, test_df['rating'].values)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplainableNeuMF(num_users, num_items, num_genres).to(device)

    model_path = os.path.join(models_dir, 'neumf_model_small.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    actuals = []
    predictions = []

    print("Running predictions")

    with torch.no_grad(): 
        # Unpack the 4 elements during testing as well
        for users, items, genres, ratings in test_loader:
            users = users.to(device)
            items = items.to(device)
            genres = genres.to(device)
            
            # Predict using all 3 features
            preds = model(users, items, genres).cpu().numpy()

            predictions.extend(preds)
            actuals.extend(ratings.numpy())

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print("\n========================================")
    print("EVALUATION RESULTS")
    print("========================================")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print("========================================\n")
