import numpy as np
import pandas as pd
from scipy import sparse

from src.evaluation import (
    _ensure_genre_features,
    _genre_columns,
    _load_saved_genre_columns,
    load_master_data,
)


class ItemKNNRecommender:
    """
    Item-item collaborative filtering recommender using sparse cosine similarity.

    Recommendations are explained by the user's previously liked movies that are
    most similar to each recommendation.
    """

    def __init__(self, min_positive_rating=4.0, top_history_items=50):
        self.min_positive_rating = min_positive_rating
        self.top_history_items = top_history_items

        self.df = load_master_data()
        self.genre_cols = _load_saved_genre_columns() or _genre_columns(self.df)
        self.df = _ensure_genre_features(self.df, self.genre_cols)

        self.movies = (
            self.df.drop_duplicates(subset=["movieId"])[["movieId", "title"] + self.genre_cols]
            .set_index("movieId")
            .sort_index()
        )
        self.train_df = self._prepare_training_data(self.df)
        self._build_sparse_similarity_inputs()

    def _prepare_training_data(self, df):
        df = df.sort_values(["userId", "datetime"]).copy()
        df["rank"] = df.groupby("userId").cumcount()
        df["count"] = df.groupby("userId")["userId"].transform("count")
        return df[df["rank"] < df["count"] * 0.8].copy()

    def _build_sparse_similarity_inputs(self):
        positive_df = self.train_df[self.train_df["rating"] >= self.min_positive_rating].copy()
        if positive_df.empty:
            raise ValueError("No positive interactions available for item-item KNN.")

        self.item_ids = np.array(sorted(self.train_df["movieId"].unique()))
        self.user_ids = np.array(sorted(self.train_df["userId"].unique()))
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

        rows = positive_df["movieId"].map(self.item_to_idx).to_numpy()
        cols = positive_df["userId"].map(self.user_to_idx).to_numpy()
        values = np.ones(len(positive_df), dtype=np.float32)

        matrix = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.item_ids), len(self.user_ids)),
        )
        row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        row_norms[row_norms == 0] = 1.0
        self.item_user_matrix = sparse.diags(1.0 / row_norms).dot(matrix).tocsr()

        self.user_positive_items = (
            positive_df.sort_values(["userId", "rating"], ascending=[True, False])
            .groupby("userId")["movieId"]
            .apply(list)
            .to_dict()
        )
        self.user_seen_items = self.train_df.groupby("userId")["movieId"].apply(set).to_dict()

    def available_user_ids(self):
        return sorted(self.train_df["userId"].unique().tolist())

    def user_liked_movies(self, user_id, limit=10):
        user_df = self.train_df[
            (self.train_df["userId"] == int(user_id))
            & (self.train_df["rating"] >= self.min_positive_rating)
        ].sort_values(["rating", "datetime"], ascending=[False, False])

        return user_df[["movieId", "title", "rating"]].head(limit).reset_index(drop=True)

    def _score_candidates(self, user_id, candidate_ids):
        scores = {item: 0.0 for item in candidate_ids}
        evidence = {item: [] for item in candidate_ids}

        history = [
            item
            for item in self.user_positive_items.get(int(user_id), [])[: self.top_history_items]
            if item in self.item_to_idx
        ]
        candidate_ids = [item for item in candidate_ids if item in self.item_to_idx]

        if not history or not candidate_ids:
            return scores, evidence

        candidate_indices = [self.item_to_idx[item] for item in candidate_ids]
        history_indices = [self.item_to_idx[item] for item in history]
        similarities = (
            self.item_user_matrix[candidate_indices]
            .dot(self.item_user_matrix[history_indices].transpose())
            .toarray()
        )

        for row_idx, item in enumerate(candidate_ids):
            row = similarities[row_idx]
            scores[item] = float(row.max()) if row.size else 0.0
            top_history_positions = row.argsort()[-3:][::-1]
            evidence[item] = [
                {
                    "movieId": history[pos],
                    "title": self.movies.loc[history[pos], "title"],
                    "similarity": float(row[pos]),
                }
                for pos in top_history_positions
                if row[pos] > 0 and history[pos] in self.movies.index
            ]

        return scores, evidence

    def recommend_for_user(self, user_id, top_n=10, candidate_pool_size=None):
        user_id = int(user_id)
        if user_id not in self.user_to_idx:
            return pd.DataFrame()

        seen_items = self.user_seen_items.get(user_id, set())
        candidates = [item for item in self.item_ids.tolist() if item not in seen_items]
        if candidate_pool_size is not None:
            candidates = candidates[:candidate_pool_size]

        scores, evidence = self._score_candidates(user_id, candidates)
        ranked_items = sorted(candidates, key=lambda item: scores[item], reverse=True)[:top_n]
        return self._format_recommendations(ranked_items, scores, evidence)

    def recommend_similar_movies(self, movie_title, top_n=10):
        matches = self.movies[
            self.movies["title"].str.contains(movie_title, case=False, na=False)
        ]
        if matches.empty:
            return pd.DataFrame()

        seed_movie_id = int(matches.index[0])
        if seed_movie_id not in self.item_to_idx:
            return pd.DataFrame()

        seed_idx = self.item_to_idx[seed_movie_id]
        candidate_ids = [item for item in self.item_ids.tolist() if item != seed_movie_id]
        candidate_indices = [self.item_to_idx[item] for item in candidate_ids]
        similarities = (
            self.item_user_matrix[candidate_indices]
            .dot(self.item_user_matrix[seed_idx].transpose())
            .toarray()
            .ravel()
        )
        scores = dict(zip(candidate_ids, similarities))
        ranked_items = sorted(candidate_ids, key=lambda item: scores[item], reverse=True)[:top_n]
        evidence = {
            item: [
                {
                    "movieId": seed_movie_id,
                    "title": self.movies.loc[seed_movie_id, "title"],
                    "similarity": float(scores[item]),
                }
            ]
            for item in ranked_items
        }
        return self._format_recommendations(ranked_items, scores, evidence)

    def _format_recommendations(self, ranked_items, scores, evidence):
        rows = []
        for item in ranked_items:
            if item not in self.movies.index:
                continue

            movie_row = self.movies.loc[item]
            genres = [genre for genre in self.genre_cols if float(movie_row[genre]) > 0]
            rows.append(
                {
                    "movieId": item,
                    "title": movie_row["title"],
                    "genres": " | ".join(genres) if genres else "(no genres listed)",
                    "relevance_score": round(float(scores[item]), 4),
                    "explanations": evidence.get(item, []),
                }
            )

        return pd.DataFrame(rows)


if __name__ == "__main__":
    engine = ItemKNNRecommender()
    print(engine.recommend_for_user(user_id=1, top_n=5))
