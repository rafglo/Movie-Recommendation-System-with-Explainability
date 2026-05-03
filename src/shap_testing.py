# Importing required packages
import pandas as pd
import torch
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt

from src.neural_cf import ExplainableNeuMF, prepare_split, safe_transform

print("Loading data...")

df = pd.read_parquet("data/processed/master_data_small.parquet")
train_df, test_df = prepare_split(df)

test_sample = test_df.sample(200, random_state=42)

print("Loading encoders...")

with open("models/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("models/item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

with open("models/genre_cols.pkl", "rb") as f:
    genre_cols = pickle.load(f)

# encode users/items
users = safe_transform(user_encoder, test_sample['userId'])
items = safe_transform(item_encoder, test_sample['movieId'])

# genres matrix (0/1 features)
genres = test_sample[genre_cols].values

# final input
X = np.column_stack((users, items, genres))

print("Loading model...")

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
num_genres = len(genre_cols)

model = ExplainableNeuMF(num_users, num_items, num_genres)

model.load_state_dict(
    torch.load("models/neumf_model_small.pth", map_location="cpu")
)
model.eval()

print("Testing prediction...")


def model_predict(x):
    x = torch.tensor(x, dtype=torch.float32)

    users = x[:, 0].long()
    items = x[:, 1].long()
    genres = x[:, 2:]

    with torch.no_grad():
        preds = model(users, items, genres).numpy()

    return preds


print(model_predict(X[:5]))

print("Running SHAP...")

feature_names = ["user", "item"] + genre_cols

explainer = shap.Explainer(model_predict, X[:50],feature_names=feature_names)
shap_values = explainer(X[:100])


actual_ratings = test_df["rating"]

