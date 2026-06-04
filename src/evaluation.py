import os
import pickle
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd


def safe_transform(encoder, values):
    mapping = {k: i + 1 for i, k in enumerate(encoder.classes_)}
    return np.array([mapping.get(v, 0) for v in values])


def prepare_split(df):
    df = df.sort_values(["userId", "datetime"])
    df["rank"] = df.groupby("userId").cumcount()
    df["count"] = df.groupby("userId")["userId"].transform("count")

    train_df = df[df["rank"] < df["count"] * 0.8].copy()
    test_df = df[df["rank"] >= df["count"] * 0.8].copy()

    return train_df, test_df


def _project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def load_master_data():
    """
    Loads the prepared interaction table. Falls back to raw CSV files when
    Parquet support is missing in the active Python environment.
    """
    root = _project_root()
    processed_path = os.path.join(root, "data", "processed", "master_data_small.parquet")

    try:
        return pd.read_parquet(processed_path)
    except (ImportError, FileNotFoundError):
        raw_dir = os.path.join(root, "data", "raw")
        ratings = pd.read_csv(os.path.join(raw_dir, "ratings.csv"))
        movies = pd.read_csv(os.path.join(raw_dir, "movies.csv"))

        ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
        genre_dummies = movies["genres"].str.get_dummies(sep="|")
        movies_with_genres = pd.concat([movies, genre_dummies], axis=1)
        cols_to_merge = ["movieId", "title"] + list(genre_dummies.columns)

        master_df = ratings.merge(
            movies_with_genres[cols_to_merge],
            on="movieId",
            how="left",
        )
        return master_df.sort_values(["userId", "datetime"])


def _genre_columns(df):
    standard_cols = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
        "datetime",
        "title",
        "rank",
        "count",
    }
    return [col for col in df.columns if col not in standard_cols]


def _load_saved_genre_columns():
    genre_path = os.path.join(_project_root(), "models", "genre_cols.pkl")
    if not os.path.exists(genre_path):
        return None

    with open(genre_path, "rb") as f:
        return pickle.load(f)


def _ensure_genre_features(df, genre_cols):
    """
    Ensures the interaction table has the one-hot genre columns expected by
    the trained model. This repairs older processed datasets that only contain
    ratings and titles.
    """
    if not genre_cols:
        return df

    missing_cols = [col for col in genre_cols if col not in df.columns]
    if not missing_cols:
        return df

    raw_movies_path = os.path.join(_project_root(), "data", "raw", "movies.csv")
    movies = pd.read_csv(raw_movies_path)
    genre_dummies = movies["genres"].str.get_dummies(sep="|")

    for col in genre_cols:
        if col not in genre_dummies.columns:
            genre_dummies[col] = 0

    genre_lookup = pd.concat(
        [movies[["movieId"]], genre_dummies[genre_cols]],
        axis=1,
    )

    df_without_stale_genres = df.drop(
        columns=[col for col in genre_cols if col in df.columns],
        errors="ignore",
    )
    repaired_df = df_without_stale_genres.merge(genre_lookup, on="movieId", how="left")
    repaired_df[genre_cols] = repaired_df[genre_cols].fillna(0).astype(np.float32)

    print(
        "Added missing genre features from raw movie metadata: "
        f"{len(missing_cols)} columns."
    )
    return repaired_df


def _dcg_at_k(relevance, k):
    relevance = np.asarray(relevance, dtype=np.float32)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance / discounts))


def ranking_metrics(ranked_items, positive_items, k):
    positive_items = set(positive_items)
    top_k = list(ranked_items[:k])

    if not positive_items:
        return None

    hits = [1 if item in positive_items else 0 for item in top_k]
    hit_count = sum(hits)
    ideal_hits = min(len(positive_items), k)
    ideal_relevance = [1] * ideal_hits
    idcg = _dcg_at_k(ideal_relevance, k)

    return {
        f"precision@{k}": hit_count / k,
        f"recall@{k}": hit_count / len(positive_items),
        f"hit_rate@{k}": 1.0 if hit_count > 0 else 0.0,
        f"ndcg@{k}": _dcg_at_k(hits, k) / idcg if idcg > 0 else 0.0,
    }


def _build_user_sets(train_df, test_df, min_positive_rating):
    train_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    positives = (
        test_df[test_df["rating"] >= min_positive_rating]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    return train_seen, positives


def _sample_users(positives_by_user, max_users, random_state):
    users = sorted(positives_by_user.keys())
    if max_users is None or len(users) <= max_users:
        return users
    rng = np.random.default_rng(random_state)
    return sorted(rng.choice(users, size=max_users, replace=False).tolist())


def _candidate_items(user_id, all_items, train_seen, positives, negatives_per_user, rng):
    seen = train_seen.get(user_id, set())
    positive_items = positives.get(user_id, set())
    negative_pool = np.array(list(all_items - seen - positive_items))

    if negatives_per_user is None or len(negative_pool) <= negatives_per_user:
        sampled_negatives = set(negative_pool.tolist())
    else:
        sampled_negatives = set(
            rng.choice(negative_pool, size=negatives_per_user, replace=False).tolist()
        )

    return list(positive_items | sampled_negatives)


def _popularity_scores(train_df):
    stats = train_df.groupby("movieId")["rating"].agg(["mean", "count"])
    global_mean = train_df["rating"].mean()
    smoothing = 10
    stats["score"] = (
        (stats["mean"] * stats["count"] + global_mean * smoothing)
        / (stats["count"] + smoothing)
    )
    return stats["score"].to_dict()


class GenrePersonalizedPopularity:
    def __init__(self, train_df, movie_lookup, genre_cols, popularity_scores, global_score):
        self.movie_lookup = movie_lookup
        self.genre_cols = genre_cols
        self.popularity_scores = popularity_scores
        self.global_score = global_score
        self.user_profiles = self._build_user_profiles(train_df)

    def _build_user_profiles(self, train_df):
        profiles = {}
        for user_id, user_df in train_df.groupby("userId"):
            liked_df = user_df[user_df["rating"] >= 4.0]
            if liked_df.empty:
                liked_df = user_df

            genre_matrix = liked_df[self.genre_cols].to_numpy(dtype=np.float32)
            weights = liked_df["rating"].to_numpy(dtype=np.float32)
            weights = np.maximum(weights - weights.mean(), 0.25)
            profile = np.average(genre_matrix, axis=0, weights=weights)
            norm = np.linalg.norm(profile)
            profiles[user_id] = profile / norm if norm > 0 else profile

        return profiles

    def score(self, user_id, candidates):
        profile = self.user_profiles.get(user_id)
        scores = {}

        for item in candidates:
            base_score = self.popularity_scores.get(item, self.global_score)
            if profile is None or item not in self.movie_lookup.index:
                scores[item] = base_score
                continue

            item_genres = self.movie_lookup.loc[item, self.genre_cols].to_numpy(dtype=np.float32)
            item_norm = np.linalg.norm(item_genres)
            genre_score = float(np.dot(profile, item_genres) / item_norm) if item_norm > 0 else 0.0
            scores[item] = base_score + genre_score

        return scores


class ItemItemKNNScorer:
    def __init__(self, train_df, min_positive_rating=4.0, top_history_items=50):
        self.min_positive_rating = min_positive_rating
        self.top_history_items = top_history_items
        self.available = False

        try:
            from scipy import sparse
        except ModuleNotFoundError:
            print("SciPy is not installed; skipping item-item KNN evaluation.")
            return

        positive_df = train_df[train_df["rating"] >= min_positive_rating].copy()
        if positive_df.empty:
            print("No positive training interactions; skipping item-item KNN evaluation.")
            return

        self.item_ids = np.array(sorted(train_df["movieId"].unique()))
        self.user_ids = np.array(sorted(train_df["userId"].unique()))
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
        self.available = True

    def score(self, user_id, candidates):
        scores = {item: 0.0 for item in candidates}
        if not self.available:
            return scores

        history = [
            item for item in self.user_positive_items.get(user_id, [])[: self.top_history_items]
            if item in self.item_to_idx
        ]
        candidate_ids = [item for item in candidates if item in self.item_to_idx]

        if not history or not candidate_ids:
            return scores

        candidate_indices = [self.item_to_idx[item] for item in candidate_ids]
        history_indices = [self.item_to_idx[item] for item in history]

        similarities = (
            self.item_user_matrix[candidate_indices]
            .dot(self.item_user_matrix[history_indices].transpose())
            .toarray()
        )
        candidate_scores = similarities.max(axis=1)

        for item, score in zip(candidate_ids, candidate_scores):
            scores[item] = float(score)

        return scores


def _load_neumf_model(df, genre_cols):
    try:
        import torch
        from src.neural_cf import ExplainableNeuMF
    except ModuleNotFoundError:
        print("PyTorch is not installed; skipping NeuMF ranking evaluation.")
        return None

    root = _project_root()
    models_dir = os.path.join(root, "models")
    model_path = os.path.join(models_dir, "neumf_model_small.pth")

    if not os.path.exists(model_path):
        return None

    with open(os.path.join(models_dir, "user_encoder.pkl"), "rb") as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(models_dir, "item_encoder.pkl"), "rb") as f:
        item_encoder = pickle.load(f)

    saved_genre_cols = _load_saved_genre_columns()
    if saved_genre_cols:
        genre_cols = saved_genre_cols

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplainableNeuMF(
        num_users=len(user_encoder.classes_),
        num_items=len(item_encoder.classes_),
        num_genres=len(genre_cols),
    ).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    movie_lookup = df.drop_duplicates(subset=["movieId"]).set_index("movieId")

    return {
        "device": device,
        "model": model,
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
        "movie_lookup": movie_lookup,
        "genre_cols": genre_cols,
    }


def _load_neumf_ranking_model(df, genre_cols):
    try:
        import torch
        from src.neural_cf import RankingNeuMF
    except ModuleNotFoundError:
        print("PyTorch is not installed; skipping ranking NeuMF evaluation.")
        return None

    root = _project_root()
    models_dir = os.path.join(root, "models")
    model_path = os.path.join(models_dir, "neumf_ranking_model.pth")

    if not os.path.exists(model_path):
        return None

    with open(os.path.join(models_dir, "ranking_user_encoder.pkl"), "rb") as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(models_dir, "ranking_item_encoder.pkl"), "rb") as f:
        item_encoder = pickle.load(f)
    with open(os.path.join(models_dir, "ranking_genre_cols.pkl"), "rb") as f:
        genre_cols = pickle.load(f)

    df = _ensure_genre_features(df, genre_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RankingNeuMF(
        num_users=len(user_encoder.classes_),
        num_items=len(item_encoder.classes_),
        num_genres=len(genre_cols),
    ).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    movie_lookup = df.drop_duplicates(subset=["movieId"]).set_index("movieId")

    return {
        "device": device,
        "model": model,
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
        "movie_lookup": movie_lookup,
        "genre_cols": genre_cols,
    }


def _score_neumf(model_bundle, user_id, candidate_ids):
    import torch

    device = model_bundle["device"]
    user_encoder = model_bundle["user_encoder"]
    item_encoder = model_bundle["item_encoder"]
    movie_lookup = model_bundle["movie_lookup"]
    genre_cols = model_bundle["genre_cols"]

    user_ids = np.array([user_id] * len(candidate_ids))
    item_ids = np.array(candidate_ids)

    user_tensor = torch.tensor(
        safe_transform(user_encoder, user_ids),
        dtype=torch.long,
    ).to(device)
    item_tensor = torch.tensor(
        safe_transform(item_encoder, item_ids),
        dtype=torch.long,
    ).to(device)
    genre_tensor = torch.tensor(
        movie_lookup.loc[item_ids, genre_cols].values,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        scores = model_bundle["model"](user_tensor, item_tensor, genre_tensor).cpu().numpy()

    return dict(zip(candidate_ids, scores))


def _format_candidate_mode(negatives_per_user):
    if negatives_per_user is None:
        return "full_catalog"
    return f"sampled_{negatives_per_user}_negatives"


def _weights_label(weights):
    return "|".join(f"{name}:{weights.get(name, 0.0):.2f}" for name in ("item_knn", "popularity", "genre"))


def _save_evaluation_outputs(
    summary,
    raw_metrics,
    run_metadata,
    save_history=True,
):
    reports_dir = os.path.join(_project_root(), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    latest_summary_path = os.path.join(reports_dir, "topk_evaluation_summary.csv")
    summary.to_csv(latest_summary_path, index=False)
    print(f"Saved summary to {latest_summary_path}")

    if not save_history:
        return latest_summary_path

    runs_dir = os.path.join(reports_dir, "evaluation_runs")
    os.makedirs(runs_dir, exist_ok=True)

    run_id = run_metadata["run_id"]
    summary_path = os.path.join(runs_dir, f"{run_id}_summary.csv")
    raw_path = os.path.join(runs_dir, f"{run_id}_per_user_metrics.csv")
    metadata_path = os.path.join(runs_dir, f"{run_id}_metadata.csv")
    leaderboard_path = os.path.join(runs_dir, "leaderboard.csv")

    summary_with_meta = summary.assign(**run_metadata)
    raw_with_meta = raw_metrics.assign(**run_metadata)
    metadata_df = pd.DataFrame([run_metadata])

    summary_with_meta.to_csv(summary_path, index=False)
    raw_with_meta.to_csv(raw_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    if os.path.exists(leaderboard_path):
        leaderboard = pd.read_csv(leaderboard_path)
        leaderboard = pd.concat([leaderboard, summary_with_meta], ignore_index=True)
    else:
        leaderboard = summary_with_meta
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"Saved timestamped summary to {summary_path}")
    print(f"Updated leaderboard at {leaderboard_path}")

    return summary_path


def evaluate_topk_recommenders(
    k_values=(5, 10),
    min_positive_rating=4.0,
    negatives_per_user=100,
    max_users=100,
    random_state=42,
    include_neumf=True,
    include_hybrid=True,
    hybrid_weights=None,
    diversity_weight=0.0,
    run_label=None,
    save_history=True,
):
    """
    Compares recommenders on top-K ranking metrics using a temporal split.

    Positives are held-out test interactions with rating >= min_positive_rating.
    Each user is evaluated against their positives plus sampled unrated negatives.
    """
    df = load_master_data()
    genre_cols = _load_saved_genre_columns() or _genre_columns(df)
    df = _ensure_genre_features(df, genre_cols)
    train_df, test_df = prepare_split(df)
    candidate_mode = _format_candidate_mode(negatives_per_user)

    train_seen, positives_by_user = _build_user_sets(
        train_df,
        test_df,
        min_positive_rating,
    )
    eval_users = _sample_users(positives_by_user, max_users, random_state)
    all_items = set(df["movieId"].unique())
    rng = np.random.default_rng(random_state)

    popularity = _popularity_scores(train_df)
    global_pop_score = train_df["rating"].mean()
    movie_lookup = df.drop_duplicates(subset=["movieId"]).set_index("movieId")
    genre_personalized = GenrePersonalizedPopularity(
        train_df=train_df,
        movie_lookup=movie_lookup,
        genre_cols=genre_cols,
        popularity_scores=popularity,
        global_score=global_pop_score,
    )
    item_item_knn = ItemItemKNNScorer(
        train_df=train_df,
        min_positive_rating=min_positive_rating,
    )
    hybrid_ranker = None
    if include_hybrid:
        from src.hybrid_ranker import DEFAULT_HYBRID_WEIGHTS, HybridRanker

        hybrid_weights = hybrid_weights or DEFAULT_HYBRID_WEIGHTS.copy()
        hybrid_ranker = HybridRanker(
            min_positive_rating=min_positive_rating,
            weights=hybrid_weights,
            diversity_weight=diversity_weight,
        )

    recommenders = {
        "popularity_bayes": lambda user_id, candidates: {
            item: popularity.get(item, global_pop_score) for item in candidates
        },
        "genre_personalized_popularity": lambda user_id, candidates: genre_personalized.score(
            user_id,
            candidates,
        ),
        "random": lambda user_id, candidates: {
            item: score for item, score in zip(candidates, rng.random(len(candidates)))
        },
    }

    if item_item_knn.available:
        recommenders["item_item_knn"] = lambda user_id, candidates: item_item_knn.score(
            user_id,
            candidates,
        )

    if hybrid_ranker is not None:
        recommenders["hybrid_ranker"] = lambda user_id, candidates: {
            item: score
            for item, score in zip(
                hybrid_ranker.rank_items(
                    user_id,
                    candidates,
                    top_n=len(candidates),
                    weights=hybrid_weights,
                    diversity_weight=diversity_weight,
                )[0],
                range(len(candidates), 0, -1),
            )
        }

    neumf_bundle = _load_neumf_model(df, genre_cols) if include_neumf else None
    if neumf_bundle is not None:
        recommenders["neumf_rating_model"] = lambda user_id, candidates: _score_neumf(
            neumf_bundle,
            user_id,
            candidates,
        )

    ranking_neumf_bundle = _load_neumf_ranking_model(df, genre_cols) if include_neumf else None
    if ranking_neumf_bundle is not None:
        recommenders["neumf_ranking_model"] = lambda user_id, candidates: _score_neumf(
            ranking_neumf_bundle,
            user_id,
            candidates,
        )

    metric_rows = []
    evaluated_user_count = 0
    candidate_counts = []

    for user_id in eval_users:
        positives = positives_by_user[user_id]
        candidates = _candidate_items(
            user_id,
            all_items,
            train_seen,
            positives_by_user,
            negatives_per_user,
            rng,
        )

        if not candidates:
            continue

        evaluated_user_count += 1
        candidate_counts.append(len(candidates))

        for recommender_name, score_fn in recommenders.items():
            scores = score_fn(user_id, candidates)
            ranked_items = sorted(candidates, key=lambda item: scores[item], reverse=True)

            for k in k_values:
                metrics = ranking_metrics(ranked_items, positives, k)
                if metrics is None:
                    continue
                metric_rows.append(
                    {
                        "recommender": recommender_name,
                        "userId": user_id,
                        **metrics,
                    }
                )

    if not metric_rows:
        raise ValueError("No users with positive test interactions were available for evaluation.")

    raw_metrics = pd.DataFrame(metric_rows)
    metric_cols = [col for col in raw_metrics.columns if "@" in col]
    summary = raw_metrics.groupby("recommender")[metric_cols].mean().reset_index()
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_parts = ["topk", candidate_mode, run_id_timestamp]
    if run_label:
        safe_label = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in run_label.strip()
        )
        if safe_label:
            run_id_parts.insert(1, safe_label)
    run_metadata = {
        "run_id": "_".join(run_id_parts),
        "run_timestamp": run_timestamp,
        "run_label": run_label or "",
        "candidate_mode": candidate_mode,
        "negatives_per_user": "all_unseen" if negatives_per_user is None else negatives_per_user,
        "max_users": "all" if max_users is None else max_users,
        "users_evaluated": evaluated_user_count,
        "positive_threshold": min_positive_rating,
        "k_values": "|".join(str(k) for k in k_values),
        "include_neumf": include_neumf,
        "include_hybrid": include_hybrid,
        "hybrid_weights": _weights_label(hybrid_weights or {}),
        "diversity_weight": diversity_weight,
        "avg_candidates_per_user": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }

    print("\n========================================")
    print("TOP-K RANKING EVALUATION")
    print("========================================")
    print(f"Users evaluated        : {evaluated_user_count}")
    print(f"Positive threshold     : rating >= {min_positive_rating}")
    print(f"Candidate mode         : {candidate_mode}")
    print(f"Negatives per user     : {'all unseen' if negatives_per_user is None else negatives_per_user}")
    print(f"Avg candidates/user    : {run_metadata['avg_candidates_per_user']:.1f}")
    if include_hybrid:
        print(f"Hybrid weights         : {run_metadata['hybrid_weights']}")
        print(f"Diversity weight       : {diversity_weight}")
    print(f"K values               : {list(k_values)}")
    print("----------------------------------------")
    print(summary.round(4).to_string(index=False))
    print("========================================\n")

    _save_evaluation_outputs(
        summary=summary,
        raw_metrics=raw_metrics,
        run_metadata=run_metadata,
        save_history=save_history,
    )

    return summary


def tune_hybrid_weights(
    k=10,
    weight_step=0.25,
    diversity_values=(0.0, 0.05, 0.10),
    min_positive_rating=4.0,
    negatives_per_user=100,
    max_users=100,
    random_state=42,
):
    """
    Grid-searches hybrid weights and diversity strength using NDCG@K.
    """
    from src.hybrid_ranker import HybridRanker

    df = load_master_data()
    genre_cols = _load_saved_genre_columns() or _genre_columns(df)
    df = _ensure_genre_features(df, genre_cols)
    train_df, test_df = prepare_split(df)
    train_seen, positives_by_user = _build_user_sets(
        train_df,
        test_df,
        min_positive_rating,
    )
    eval_users = _sample_users(positives_by_user, max_users, random_state)
    all_items = set(df["movieId"].unique())
    rng = np.random.default_rng(random_state)
    ranker = HybridRanker(min_positive_rating=min_positive_rating)
    user_candidates = {
        user_id: _candidate_items(
            user_id,
            all_items,
            train_seen,
            positives_by_user,
            negatives_per_user,
            rng,
        )
        for user_id in eval_users
    }
    user_components = {
        user_id: ranker.component_scores(user_id, user_candidates[user_id])
        for user_id in eval_users
    }

    values = np.arange(0.0, 1.0 + weight_step, weight_step)
    weight_sets = []
    for item_knn_weight, popularity_weight, genre_weight in product(values, repeat=3):
        total = item_knn_weight + popularity_weight + genre_weight
        if np.isclose(total, 1.0):
            weight_sets.append(
                {
                    "item_knn": float(item_knn_weight),
                    "popularity": float(popularity_weight),
                    "genre": float(genre_weight),
                }
            )

    rows = []
    for weights in weight_sets:
        for diversity_weight in diversity_values:
            user_metrics = []

            for user_id in eval_users:
                positives = positives_by_user[user_id]
                candidates = user_candidates[user_id]
                components = user_components[user_id]
                weight_sum = sum(weights.values())
                scores = {
                    item: sum(
                        (weights[name] / weight_sum) * components[name].get(item, 0.0)
                        for name in ("item_knn", "popularity", "genre")
                    )
                    for item in candidates
                }
                if diversity_weight > 0:
                    ranked_items = ranker._diversity_rerank(
                        candidates,
                        scores,
                        k,
                        diversity_weight,
                    )
                else:
                    ranked_items = sorted(
                        candidates,
                        key=lambda item: scores[item],
                        reverse=True,
                    )[:k]
                metrics = ranking_metrics(ranked_items, positives, k)
                if metrics is not None:
                    user_metrics.append(metrics)

            if not user_metrics:
                continue

            metrics_df = pd.DataFrame(user_metrics)
            row = {
                "item_knn_weight": weights["item_knn"],
                "popularity_weight": weights["popularity"],
                "genre_weight": weights["genre"],
                "diversity_weight": diversity_weight,
                "users_evaluated": len(user_metrics),
                f"precision@{k}": metrics_df[f"precision@{k}"].mean(),
                f"recall@{k}": metrics_df[f"recall@{k}"].mean(),
                f"hit_rate@{k}": metrics_df[f"hit_rate@{k}"].mean(),
                f"ndcg@{k}": metrics_df[f"ndcg@{k}"].mean(),
            }
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(f"ndcg@{k}", ascending=False)
    reports_dir = os.path.join(_project_root(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.join(reports_dir, "hybrid_weight_tuning.csv")
    results.to_csv(output_path, index=False)

    print("\n========================================")
    print("HYBRID WEIGHT TUNING")
    print("========================================")
    print(f"K value                : {k}")
    print(f"Weight step            : {weight_step}")
    print(f"Diversity values       : {list(diversity_values)}")
    print(f"Users evaluated        : {max_users if max_users is not None else 'all'}")
    print("----------------------------------------")
    print(results.head(10).round(4).to_string(index=False))
    print("========================================\n")
    print(f"Saved tuning results to {output_path}")

    return results


if __name__ == "__main__":
    evaluate_topk_recommenders()
