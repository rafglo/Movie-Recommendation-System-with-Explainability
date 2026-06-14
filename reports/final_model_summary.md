# Final Model Summary

## Objective

Build a movie recommendation system that produces accurate top-K
recommendations and explains why each movie was recommended.

## Evaluation Protocol

Main benchmark:

- temporal per-user split
- held-out positives: rating >= 4.0
- sampled negatives: 100 unseen movies per user
- users evaluated: 100
- primary metric: NDCG@10

Final training and evaluation are logged in MLflow:

```text
movie_recommender_training
movie_recommender_evaluation
```

Open the MLflow UI with:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Full-catalog evaluation is also available with:

```powershell
python main.py --mode eval_topk --full-catalog
```

## Final Sampled Results

| Recommender | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| hybrid ranker | 0.397 | 0.504 | 0.950 | 0.584 |
| ranking NeuMF | 0.415 | 0.485 | 0.910 | 0.569 |
| item-item KNN | 0.382 | 0.483 | 0.940 | 0.541 |
| popularity baseline | 0.330 | 0.381 | 0.850 | 0.473 |
| rating NeuMF | 0.176 | 0.182 | 0.710 | 0.198 |
| random baseline | 0.104 | 0.120 | 0.620 | 0.136 |

## Selected GUI Model

The Streamlit app uses the hybrid ranker.

Reason:

- it is the strongest model in the final logged sampled benchmark
- it has stronger direct explainability
- it exposes clear component scores: item similarity, popularity, and genre match

## Neural Model Improvement

The original NeuMF was trained as a rating predictor and performed poorly for
top-K recommendations. Retraining NeuMF as a ranking model fixed the objective
mismatch.

Ranking NeuMF changes:

- positives: ratings >= 4.0
- negatives: sampled unrated movies
- loss: BCEWithLogitsLoss
- output: raw relevance logit
- validation: Precision@10, Recall@10, HitRate@10, NDCG@10

Training history:

| Epoch | Loss | NDCG@10 |
| ---: | ---: | ---: |
| 1 | 0.585 | 0.175 |
| 2 | 0.461 | 0.332 |
| 3 | 0.419 | 0.457 |
| 4 | 0.377 | 0.537 |
| 5 | 0.338 | 0.591 |
| 6 | 0.311 | 0.594 |
| 7 | 0.296 | 0.603 |
| 8 | 0.285 | 0.606 |
| 9 | 0.278 | 0.601 |
| 10 | 0.272 | 0.605 |

The final ranking NeuMF training run logs:

- training parameters
- per-epoch loss and top-K validation metrics
- best model artifact
- encoders and genre metadata
- training history CSV

The final `eval_topk` run logs:

- benchmark parameters
- per-recommender Precision, Recall, HitRate, and NDCG
- summary CSV
- per-user metrics CSV
- leaderboard CSV

## Explainability Strategy

Model-specific explanations are used first:

- item-item KNN: recommended because it is similar to movies the user liked
- hybrid ranker: exact weighted component contributions
- ranking NeuMF: genre occlusion sensitivity

SHAP is used in two layers:

Faithful hybrid-level SHAP:

- item KNN score
- popularity score
- genre score

Original-feature surrogate SHAP:

- genres
- tags
- rating statistics
- user profile features
- collaborative evidence features

The hybrid-level SHAP values explain the deployed scoring formula directly.
The original-feature SHAP analysis explains surrogate models trained to
approximate each component score, so it should be interpreted as feature-level
insight rather than the exact internal formula of item-item KNN.

## Next Recommended Work

- add screenshots to the final presentation or report
- optionally add a ranking NeuMF app mode
- evaluate all models on a larger full-catalog user sample
- add poster images through `links.csv`/TMDB or another image source
- clean old exploratory run artifacts before final submission
