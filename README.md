# Movie Recommendation System with Explainability

This project builds and evaluates an explainable movie recommendation system on
MovieLens data. It includes collaborative filtering, neural collaborative
filtering, hybrid ranking, top-K evaluation, and a Streamlit application.

## Current Status

The project now has a working model pipeline and GUI.

- Best benchmark model: hybrid ranker
- Best explainable GUI model: hybrid ranker
- GUI framework: Streamlit
- Main benchmark: top-K recommendation quality with NDCG@10

The app currently uses the hybrid ranker because it is the strongest final
logged sampled benchmark model and gives clearer recommendation explanations.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project expects the MovieLens CSV files under:

```text
data/raw/ratings.csv
data/raw/movies.csv
data/raw/tags.csv
data/raw/links.csv
```

## Run The App

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The app supports:

- user-based recommendations
- similar-movie recommendations
- relevance scores
- similar-liked-movie explanations
- KNN, popularity, and genre contribution scores
- SHAP-style structured feature attributions for the hybrid ranker
- a diversity slider

## Pipeline Commands

Prepare data:

```powershell
python main.py --mode data
```

Train the original rating-prediction NeuMF:

```powershell
python main.py --mode train_cf
```

Train the ranking NeuMF:

```powershell
python main.py --mode train_rank_neumf --ranking-epochs 5 --negative-samples 4
```

Run the standard sampled top-K benchmark:

```powershell
python main.py --mode eval_topk --run-label benchmark
```

Open the MLflow UI:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Final training and evaluation runs are logged under:

```text
movie_recommender_training
movie_recommender_evaluation
```

Run a full-catalog smoke benchmark:

```powershell
python main.py --mode eval_topk --full-catalog --max-users 25 --run-label full_catalog_smoke
```

Tune hybrid weights:

```powershell
python main.py --mode tune_hybrid --weight-step 0.25 --diversity-values 0 0.05 0.1
```

## Final Sampled Benchmark

The latest sampled benchmark is saved in:

```text
reports/topk_evaluation_summary.csv
```

Current key results:

| Recommender | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| hybrid ranker | 0.397 | 0.504 | 0.950 | 0.584 |
| ranking NeuMF | 0.415 | 0.485 | 0.910 | 0.569 |
| item-item KNN | 0.382 | 0.483 | 0.940 | 0.541 |
| popularity baseline | 0.330 | 0.381 | 0.850 | 0.473 |
| rating NeuMF | 0.195 | 0.210 | 0.760 | 0.241 |
| random | 0.104 | 0.120 | 0.620 | 0.136 |

## Why Two Strong Models?

The hybrid ranker is the strongest final logged sampled benchmark model and is
selected for the GUI because it is directly explainable. It combines:

- item-item KNN similarity
- Bayesian-smoothed popularity
- genre affinity
- optional diversity reranking

The ranking NeuMF is the strongest neural model. It is trained with positive
interactions and sampled negatives using `BCEWithLogitsLoss`.

## Explainability

The project uses model-specific explanations first:

- item-item KNN: similar movies the user liked
- hybrid ranker: weighted component contributions
- ranking NeuMF: genre occlusion utility for neural score sensitivity

SHAP is only applied to final structured hybrid features:

- item KNN score
- popularity score
- genre score

This keeps explanations interpretable instead of applying SHAP directly to
latent embedding dimensions.

## Important Artifacts

Models:

```text
models/neumf_ranking_model.pth
models/neumf_model_small.pth
models/ranking_user_encoder.pkl
models/ranking_item_encoder.pkl
models/ranking_genre_cols.pkl
```

Reports:

```text
reports/topk_evaluation_summary.csv
reports/evaluation_runs/leaderboard.csv
reports/hybrid_weight_tuning.csv
reports/neumf_ranking_training_history.csv
reports/final_model_summary.md
```

MLflow tracking:

```text
mlflow.db
mlruns/
```

## Notes

Generated files under `reports/evaluation_runs/` are useful for experiment
tracking. Keep the leaderboard and selected run summaries for the final report;
old exploratory runs can be removed if the repository becomes too noisy.
