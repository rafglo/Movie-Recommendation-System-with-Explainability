import numpy as np
import pandas as pd

from src.explainability import (
    explanation_sentence,
    hybrid_component_contributions,
    shap_hybrid_values,
)
from src.item_knn_engine import ItemKNNRecommender


DEFAULT_HYBRID_WEIGHTS = {
    "item_knn": 0.50,
    "popularity": 0.25,
    "genre": 0.25,
}


class HybridRanker:
    """
    Weighted hybrid recommender combining collaborative, popularity, and genre
    affinity signals with optional diversity-aware reranking.
    """

    def __init__(
        self,
        min_positive_rating=4.0,
        top_history_items=50,
        weights=None,
        diversity_weight=0.0,
    ):
        self.item_engine = ItemKNNRecommender(
            min_positive_rating=min_positive_rating,
            top_history_items=top_history_items,
        )
        self.min_positive_rating = min_positive_rating
        self.weights = weights or DEFAULT_HYBRID_WEIGHTS.copy()
        self.diversity_weight = diversity_weight
        self.popularity_scores = self._build_popularity_scores()
        self.global_popularity = self.item_engine.train_df["rating"].mean()
        self.genre_vectors = self._build_genre_vectors()
        self.user_genre_profiles = self._build_user_genre_profiles()

    def available_user_ids(self):
        return self.item_engine.available_user_ids()

    def user_liked_movies(self, user_id, limit=10):
        return self.item_engine.user_liked_movies(user_id, limit=limit)

    def _build_popularity_scores(self):
        stats = self.item_engine.train_df.groupby("movieId")["rating"].agg(["mean", "count"])
        global_mean = self.item_engine.train_df["rating"].mean()
        smoothing = 10
        score = (
            (stats["mean"] * stats["count"] + global_mean * smoothing)
            / (stats["count"] + smoothing)
        )
        return score.to_dict()

    def _build_genre_vectors(self):
        vectors = {}
        for item, row in self.item_engine.movies.iterrows():
            vector = row[self.item_engine.genre_cols].to_numpy(dtype=np.float32)
            norm = np.linalg.norm(vector)
            vectors[item] = vector / norm if norm > 0 else vector
        return vectors

    def _build_user_genre_profiles(self):
        profiles = {}
        genre_cols = self.item_engine.genre_cols

        for user_id, user_df in self.item_engine.train_df.groupby("userId"):
            liked_df = user_df[user_df["rating"] >= self.min_positive_rating]
            if liked_df.empty:
                liked_df = user_df

            genre_matrix = liked_df[genre_cols].to_numpy(dtype=np.float32)
            weights = liked_df["rating"].to_numpy(dtype=np.float32)
            weights = np.maximum(weights - weights.mean(), 0.25)
            profile = np.average(genre_matrix, axis=0, weights=weights)
            norm = np.linalg.norm(profile)
            profiles[user_id] = profile / norm if norm > 0 else profile

        return profiles

    def _normalize_component(self, scores, candidate_ids):
        values = np.array([scores.get(item, 0.0) for item in candidate_ids], dtype=np.float32)
        min_value = float(values.min()) if values.size else 0.0
        max_value = float(values.max()) if values.size else 0.0

        if max_value == min_value:
            return {item: 0.0 for item in candidate_ids}

        return {
            item: float((scores.get(item, 0.0) - min_value) / (max_value - min_value))
            for item in candidate_ids
        }

    def _genre_scores(self, user_id, candidate_ids):
        profile = self.user_genre_profiles.get(int(user_id))
        scores = {}

        for item in candidate_ids:
            if profile is None or item not in self.item_engine.movies.index:
                scores[item] = 0.0
                continue

            item_genres = self.item_engine.movies.loc[
                item,
                self.item_engine.genre_cols,
            ].to_numpy(dtype=np.float32)
            item_norm = np.linalg.norm(item_genres)
            scores[item] = float(np.dot(profile, item_genres) / item_norm) if item_norm > 0 else 0.0

        return scores

    def component_scores(self, user_id, candidate_ids):
        item_scores, evidence = self.item_engine._score_candidates(user_id, candidate_ids)
        popularity_scores = {
            item: self.popularity_scores.get(item, self.global_popularity)
            for item in candidate_ids
        }
        genre_scores = self._genre_scores(user_id, candidate_ids)

        return {
            "item_knn": self._normalize_component(item_scores, candidate_ids),
            "popularity": self._normalize_component(popularity_scores, candidate_ids),
            "genre": self._normalize_component(genre_scores, candidate_ids),
            "evidence": evidence,
        }

    def score_candidates(self, user_id, candidate_ids, weights=None):
        weights = weights or self.weights
        weight_sum = sum(max(value, 0.0) for value in weights.values())
        if weight_sum <= 0:
            raise ValueError("Hybrid weights must contain at least one positive value.")

        components = self.component_scores(user_id, candidate_ids)
        combined = {}

        for item in candidate_ids:
            combined[item] = sum(
                (max(weights.get(name, 0.0), 0.0) / weight_sum)
                * components[name].get(item, 0.0)
                for name in ("item_knn", "popularity", "genre")
            )

        return combined, components

    def rank_items(self, user_id, candidate_ids, top_n=10, weights=None, diversity_weight=None):
        diversity_weight = self.diversity_weight if diversity_weight is None else diversity_weight
        scores, components = self.score_candidates(user_id, candidate_ids, weights=weights)

        if diversity_weight <= 0:
            ranked = sorted(candidate_ids, key=lambda item: scores[item], reverse=True)[:top_n]
            return ranked, scores, components

        ranked = self._diversity_rerank(candidate_ids, scores, top_n, diversity_weight)
        return ranked, scores, components

    def _diversity_rerank(self, candidate_ids, scores, top_n, diversity_weight):
        remaining = set(candidate_ids)
        selected = []

        while remaining and len(selected) < top_n:
            best_item = max(
                remaining,
                key=lambda item: scores[item]
                - diversity_weight * self._max_genre_similarity(item, selected),
            )
            selected.append(best_item)
            remaining.remove(best_item)

        return selected

    def _max_genre_similarity(self, item, selected_items):
        if not selected_items or item not in self.genre_vectors:
            return 0.0

        item_vector = self.genre_vectors[item]
        if not item_vector.any():
            return 0.0

        max_similarity = 0.0
        for selected_item in selected_items:
            if selected_item not in self.genre_vectors:
                continue
            selected_vector = self.genre_vectors[selected_item]
            if not selected_vector.any():
                continue
            similarity = float(np.dot(item_vector, selected_vector))
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def recommend_for_user(self, user_id, top_n=10, weights=None, diversity_weight=None):
        user_id = int(user_id)
        if user_id not in self.item_engine.user_to_idx:
            return pd.DataFrame()

        seen_items = self.item_engine.user_seen_items.get(user_id, set())
        candidate_ids = [
            item for item in self.item_engine.item_ids.tolist()
            if item not in seen_items
        ]
        ranked, scores, components = self.rank_items(
            user_id,
            candidate_ids,
            top_n=top_n,
            weights=weights,
            diversity_weight=diversity_weight,
        )
        return self._format_recommendations(ranked, scores, components)

    def _format_recommendations(self, ranked_items, scores, components):
        rows = []
        weights = self.weights

        for item in ranked_items:
            if item not in self.item_engine.movies.index:
                continue

            movie_row = self.item_engine.movies.loc[item]
            genres = [
                genre for genre in self.item_engine.genre_cols
                if float(movie_row[genre]) > 0
            ]
            row = {
                "movieId": item,
                "title": movie_row["title"],
                "genres": " | ".join(genres) if genres else "(no genres listed)",
                "relevance_score": round(float(scores[item]), 4),
                "item_knn_score": round(float(components["item_knn"].get(item, 0.0)), 4),
                "popularity_score": round(float(components["popularity"].get(item, 0.0)), 4),
                "genre_score": round(float(components["genre"].get(item, 0.0)), 4),
                "explanations": components["evidence"].get(item, []),
            }
            row["component_contributions"] = hybrid_component_contributions(row, weights)
            row["explanation_summary"] = explanation_sentence(row, weights)
            rows.append(row)

        result = pd.DataFrame(rows)
        if not result.empty:
            shap_values = shap_hybrid_values(result, weights)
            result = pd.concat([result.reset_index(drop=True), shap_values], axis=1)

        return result


if __name__ == "__main__":
    ranker = HybridRanker()
    print(ranker.recommend_for_user(user_id=1, top_n=5))
