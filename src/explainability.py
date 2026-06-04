import numpy as np
import pandas as pd


HYBRID_FEATURES = ["item_knn_score", "popularity_score", "genre_score"]


def normalized_weights(weights):
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        return {key: 0.0 for key in weights}
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def hybrid_component_contributions(row, weights):
    """
    Exact model-specific contribution breakdown for the weighted hybrid ranker.
    """
    weights = normalized_weights(weights)
    mapping = {
        "item_knn": "item_knn_score",
        "popularity": "popularity_score",
        "genre": "genre_score",
    }
    contributions = {}

    for component, score_col in mapping.items():
        contributions[component] = float(weights.get(component, 0.0) * row.get(score_col, 0.0))

    total = sum(contributions.values())
    return {
        "contributions": contributions,
        "total": total,
        "top_component": max(contributions, key=contributions.get) if contributions else None,
    }


def explanation_sentence(row, weights):
    breakdown = hybrid_component_contributions(row, weights)
    top_component = breakdown["top_component"]
    labels = {
        "item_knn": "similarity to movies the user liked",
        "popularity": "strong global reception",
        "genre": "genre match with the user's profile",
    }

    if top_component is None:
        return "No dominant explanation signal was found."

    return f"Main signal: {labels[top_component]}."


def exact_hybrid_shap_values(rows, weights):
    """
    SHAP-equivalent exact additive attributions for the linear hybrid formula.
    This is the fallback when the optional shap package is unavailable.
    """
    weights = normalized_weights(weights)
    values = []

    for _, row in rows.iterrows():
        values.append(
            {
                "item_knn_shap": float(weights.get("item_knn", 0.0) * row["item_knn_score"]),
                "popularity_shap": float(weights.get("popularity", 0.0) * row["popularity_score"]),
                "genre_shap": float(weights.get("genre", 0.0) * row["genre_score"]),
            }
        )

    return pd.DataFrame(values)


def shap_hybrid_values(rows, weights):
    """
    Optional SHAP explanation over final structured hybrid features.
    If shap is unavailable, returns exact additive contributions instead.
    """
    feature_frame = rows[HYBRID_FEATURES].astype(float).copy()
    weight_vector = np.array(
        [
            normalized_weights(weights).get("item_knn", 0.0),
            normalized_weights(weights).get("popularity", 0.0),
            normalized_weights(weights).get("genre", 0.0),
        ],
        dtype=np.float32,
    )

    try:
        import shap
    except ModuleNotFoundError:
        return exact_hybrid_shap_values(rows, weights)

    def predict(values):
        return np.asarray(values, dtype=np.float32).dot(weight_vector)

    background = np.zeros((1, len(HYBRID_FEATURES)), dtype=np.float32)
    explainer = shap.Explainer(predict, background, feature_names=HYBRID_FEATURES)
    shap_values = explainer(feature_frame.to_numpy(dtype=np.float32))
    values = shap_values.values

    return pd.DataFrame(
        {
            "item_knn_shap": values[:, 0],
            "popularity_shap": values[:, 1],
            "genre_shap": values[:, 2],
        }
    )


def rank_genre_occlusion_effects(model, user_tensor, item_tensor, genre_tensor, genre_cols, top_n=5):
    """
    Model-specific explanation for ranking NeuMF: measure how much the logit
    drops when each active genre feature is removed.
    """
    import torch

    model.eval()
    with torch.no_grad():
        base_score = float(model(user_tensor, item_tensor, genre_tensor).cpu().item())
        effects = []
        active_indices = torch.where(genre_tensor.squeeze() > 0)[0].cpu().numpy().tolist()

        for idx in active_indices:
            perturbed = genre_tensor.clone()
            perturbed[:, idx] = 0.0
            score = float(model(user_tensor, item_tensor, perturbed).cpu().item())
            effects.append(
                {
                    "genre": genre_cols[idx],
                    "logit_drop_when_removed": base_score - score,
                }
            )

    return sorted(effects, key=lambda row: row["logit_drop_when_removed"], reverse=True)[:top_n]
