# Movie Recommendation System with Explainability - Full Explanation

## Short Overview

This project builds an explainable movie recommendation system on MovieLens data.
The final application recommends movies through a Streamlit GUI and explains why
each recommendation was produced.

The project includes several recommendation approaches:

- popularity baseline
- genre-personalized popularity
- item-item collaborative filtering
- original rating-prediction NeuMF
- retrained ranking NeuMF
- final hybrid ranker

The final GUI uses the hybrid ranker because it is the strongest final sampled
top-K benchmark model and has the clearest explanations. The strongest neural
model is the retrained ranking NeuMF.

Current final sampled benchmark:

| Recommender | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| hybrid_ranker | 0.397 | 0.504 | 0.950 | 0.584 |
| neumf_ranking_model | 0.415 | 0.485 | 0.910 | 0.569 |
| item_item_knn | 0.382 | 0.483 | 0.940 | 0.541 |
| popularity_bayes | 0.330 | 0.381 | 0.850 | 0.473 |
| random | 0.104 | 0.120 | 0.620 | 0.136 |

The main evaluation metric is `NDCG@10`, because recommendation quality depends
not only on whether relevant movies are found, but also on whether they appear
near the top of the recommendation list.

## Bottom-To-Top Workflow

The project follows this workflow:

1. Load raw MovieLens data.
2. Prepare one processed master dataset.
3. Build several recommender models and baselines.
4. Evaluate all models with top-K ranking metrics.
5. Tune and select the best explainable model.
6. Log final training and evaluation results in MLflow.
7. Expose the final model through a Streamlit application.
8. Present the final results and explanations in `notebooks/final_submission.ipynb`.

The important point is that the project moved from rating prediction toward
ranking. A movie recommender should answer "which movies should appear in the
top 10?", not only "what exact rating will this user give this movie?". This is
why the final benchmark is top-K evaluation and why the NeuMF model was retrained
with a ranking-friendly objective.

## Data Layer

Raw files are stored in:

```text
data/raw/ratings.csv
data/raw/movies.csv
data/raw/tags.csv
data/raw/links.csv
```

The processed dataset is stored in:

```text
data/processed/master_data_small.parquet
data/processed/master_data_1M.parquet
```

The raw ratings contain user-movie interactions:

- `userId`
- `movieId`
- `rating`
- `timestamp`

The raw movies file contains movie metadata:

- `movieId`
- `title`
- `genres`

During preprocessing, genre strings are converted into one-hot genre features.
For example, a movie with genres `Action|Adventure|Sci-Fi` receives binary
columns such as `Action = 1`, `Adventure = 1`, and `Sci-Fi = 1`.

The timestamp is converted into a datetime column so the evaluation can use a
temporal split. This matters because a recommender should be evaluated on future
interactions, not randomly mixed historical interactions.

## Temporal Train-Test Split

The project uses a temporal per-user split:

- for each user, interactions are sorted by time
- earlier interactions are used for training
- later interactions are held out for testing

This is more realistic than a random split. In a real recommendation system, the
model uses what the user did in the past to recommend what they may like in the
future.

The top-K evaluation considers held-out ratings with:

```text
rating >= 4.0
```

as positive/relevant movies.

## Why Top-K Evaluation Is Used

Earlier versions of the project evaluated rating prediction with metrics such as
RMSE and MAE. Those metrics are useful if the goal is to predict exact star
ratings, but they are not enough for a recommender application.

The GUI needs ranked recommendation lists. Therefore, the final evaluation asks:

> If the system recommends the top K movies, how many of those movies are
> actually relevant for the user?

The project computes:

- `Precision@K`: fraction of recommended movies that are relevant
- `Recall@K`: fraction of relevant held-out movies recovered in the top K
- `HitRate@K`: whether at least one relevant movie appears in the top K
- `NDCG@K`: whether relevant movies appear high in the ranking

`NDCG@10` is the main metric because ranking position matters. A relevant movie
at position 1 is better than the same movie at position 10.

## Candidate Generation

For each evaluated user, the system builds a candidate set of movies:

- held-out positive movies from the test split
- sampled unseen negative movies

The standard benchmark uses 100 sampled negatives per user. Full-catalog
evaluation is also available and ranks all unseen movies, but it is slower.

Sampled evaluation is useful for fast model iteration. Full-catalog evaluation
is useful as a stricter final smoke test.

## Baseline Models

### Random Baseline

The random baseline ranks candidate movies randomly. It is not a useful
recommender, but it is important as a sanity check. Every serious model should
beat random.

### Bayesian Popularity Baseline

Popularity is a strong baseline in recommendation systems. Popular movies often
perform surprisingly well because many users like them.

This project uses Bayesian-smoothed popularity, not raw average rating. Raw
average rating can overvalue movies with very few ratings. Bayesian smoothing
combines:

- the movie's average rating
- the number of ratings for the movie
- the global average rating
- a smoothing strength

This gives stable scores to popular movies and avoids giving too much trust to
movies with only a few ratings.

### Genre-Personalized Popularity

Genre-personalized popularity improves the popularity baseline by considering
the user's genre preferences.

The model estimates which genres a user tends to like from their training
history. Candidate movies are then scored using both:

- popularity
- match with the user's genre profile

This is a simple hybrid of content and popularity signals.

## Item-Item KNN Collaborative Filtering

Item-item KNN is one of the most important models in this project.

The idea is:

> Recommend movies that are similar to movies the user already liked.

The implementation builds an item-user matrix from positive interactions. A
positive interaction is usually:

```text
rating >= 4.0
```

Each row represents a movie, each column represents a user, and the matrix marks
whether the user liked that movie.

The model then normalizes movie vectors and computes cosine similarity between
movies. If two movies are liked by many of the same users, they are considered
similar.

For a target user:

1. collect movies the user liked in the training split
2. score unseen candidate movies by similarity to those liked movies
3. rank candidates by their strongest or aggregated similarity signal
4. return the top N movies

Item-item KNN is valuable because it is:

- strong for top-K recommendation
- fast enough for the app
- easy to explain

Its explanation is direct: "this movie was recommended because it is similar to
these movies you liked."

## Original NeuMF Rating Model

The project originally included a neural collaborative filtering model called
NeuMF. NeuMF combines:

- generalized matrix factorization style embeddings
- a multilayer perceptron over user/item embeddings
- genre/content features

The first version was trained as a rating-prediction model. It predicted a
rating-like output and was evaluated with rating metrics.

This turned out to be a mismatch for the real application. A model can predict
ratings reasonably but still produce weak top-10 rankings. This is why the
rating NeuMF performs poorly in the final top-K benchmark compared with the
ranking NeuMF and hybrid ranker.

## Ranking NeuMF

The ranking NeuMF fixes the objective mismatch.

Instead of predicting exact star ratings, the model learns whether a user is
likely to interact positively with a movie.

Positive examples:

```text
rating >= 4.0
```

Negative examples:

```text
movies the user has not rated, sampled as negatives
```

The model is trained with:

```text
BCEWithLogitsLoss
```

This is a binary ranking-friendly objective. The output is a raw relevance
logit, not a sigmoid-scaled rating. During ranking, candidates are sorted by this
logit.

Training history:

| Epoch | Loss | NDCG@10 |
| ---: | ---: | ---: |
| 1 | 0.585 | 0.175 |
| 2 | 0.461 | 0.332 |
| 3 | 0.419 | 0.457 |
| 4 | 0.377 | 0.537 |
| 5 | 0.338 | 0.591 |

The ranking NeuMF is the strongest neural model in the project. Its final
sampled benchmark `NDCG@10` is about `0.569`.

## Hybrid Ranker

The final GUI model is the hybrid ranker.

It combines three signals:

```text
item_item_knn score
popularity score
genre affinity score
```

Default weights:

```text
item_item_knn: 0.50
popularity:   0.25
genre:        0.25
```

The combined score is a weighted sum of normalized component scores. A movie
gets a high final score if it is:

- similar to movies the user liked
- generally popular/reliable
- aligned with the user's preferred genres

The hybrid ranker also supports a diversity penalty. Without diversity, the top
recommendations can become too similar to each other. Diversity reranking
penalizes candidates that are too genre-similar to movies already selected in
the recommendation list.

This helps avoid recommendation lists where all movies come from the same narrow
genre cluster.

The hybrid ranker is selected for the Streamlit app because:

- it has the best final sampled `NDCG@10`
- it is easier to explain than neural embeddings
- it returns component scores for every recommendation
- it supports a diversity slider in the GUI

## Explainability Strategy

The project uses model-specific explanations first.

This is important because SHAP or LIME are not automatically meaningful for all
model types. For example, explaining latent embedding dimensions from a neural
recommender is difficult because the dimensions do not have simple human
meaning.

### Item-Item KNN Explainability

The explanation is:

> This movie is recommended because it is similar to movies the user liked.

The app can show the supporting movies and similarity values.

This is faithful because those similarities are the same evidence used by the
model to rank candidates.

### Hybrid Ranker Explainability

The hybrid ranker explains recommendations through component contributions:

- item similarity contribution
- popularity contribution
- genre contribution

For example, a recommendation can be explained as:

> Main signal: similarity to movies the user liked.

or:

> This recommendation is supported by popularity and genre match.

This is practical and understandable for a non-technical user.

### Ranking NeuMF Explainability

The ranking NeuMF is less directly interpretable because it uses embeddings.
The project therefore avoids pretending that embedding dimensions have obvious
meaning.

Instead, the explainability utility supports genre occlusion sensitivity. This
tests how the model score changes when genre features are altered. It gives a
limited but honest view of which content features influence the neural score.

### SHAP Usage

SHAP is used only for structured hybrid features:

- item KNN score
- popularity score
- genre score

This is the right place to use SHAP because those features are human-readable.
The project avoids applying SHAP directly to latent embedding dimensions, where
the result would be harder to explain responsibly.

## MLflow Tracking

Final training and evaluation are logged in MLflow.

MLflow database:

```text
mlflow.db
```

MLflow run directory:

```text
mlruns/
```

Experiments:

```text
movie_recommender_training
movie_recommender_evaluation
```

The ranking NeuMF training run logs:

- training parameters
- loss per epoch
- validation top-K metrics per epoch
- best model artifact
- training history CSV

The evaluation run logs:

- benchmark configuration
- per-model Precision@K
- per-model Recall@K
- per-model HitRate@K
- per-model NDCG@K
- summary CSV
- per-user metrics CSV
- leaderboard CSV

To open the MLflow UI:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Streamlit Application Workflow

The Streamlit app is in:

```text
app.py
```

The app loads the final hybrid ranker and exposes two main workflows:

1. Recommend movies for a selected user.
2. Find movies similar to a selected movie.

For user recommendations, the app displays:

- movie title
- genres
- relevance score
- item KNN score
- popularity score
- genre score
- explanation summary
- similar liked movies used as evidence
- SHAP-style structured feature attributions
- diversity control

The app is designed around the final selected model, not around every
experimental model. This keeps the demo focused and easier to explain.

## Main Project Files

### `main.py`

Command-line pipeline entry point.

It supports these modes:

- `data`: prepare processed data
- `train_cf`: train the original rating-prediction NeuMF
- `train_rank_neumf`: train the ranking NeuMF
- `eval_cf`: evaluate the old rating-prediction model
- `eval_topk`: run the main top-K benchmark
- `tune_hybrid`: tune hybrid ranker weights
- `test_content`: test content-based recommendations
- `all`: run the broader pipeline

Important command:

```powershell
python main.py --mode eval_topk --run-label benchmark
```

### `app.py`

Streamlit GUI for the final recommendation system.

It loads `HybridRanker`, lets the user choose recommendation settings, and
renders recommendations with explanations.

### `src/data_pipeline.py`

Prepares the master dataset from raw MovieLens files.

Responsibilities:

- load raw CSV files
- merge ratings and movie metadata
- parse genres into usable features
- create processed parquet output

### `src/evaluation.py`

Core evaluation module.

Responsibilities:

- load processed data
- create temporal train/test split
- build candidate sets
- calculate top-K metrics
- evaluate baselines
- evaluate item-item KNN
- evaluate rating NeuMF
- evaluate ranking NeuMF
- evaluate hybrid ranker
- save timestamped evaluation outputs
- update leaderboard
- log evaluation results to MLflow
- tune hybrid weights

This is one of the most important files because it defines the final benchmark.

### `src/item_knn_engine.py`

Production item-item KNN recommender.

Responsibilities:

- build item-user sparse matrix
- compute cosine similarity through normalized sparse vectors
- recommend unseen movies for a user
- recommend similar movies from a title query
- provide similar-liked-movie explanation evidence

This is both a standalone recommender and a component inside the hybrid ranker.

### `src/hybrid_ranker.py`

Final hybrid recommendation model.

Responsibilities:

- combine item-item KNN, popularity, and genre scores
- normalize component scores
- apply weighted scoring
- optionally apply diversity reranking
- format recommendations for the app
- attach explanation summaries and component contributions
- attach SHAP-style values for structured hybrid features

This is the model used by the Streamlit GUI.

### `src/neural_cf.py`

Neural collaborative filtering module.

Responsibilities:

- define the original `ExplainableNeuMF`
- define the final `RankingNeuMF`
- build negative-sampled ranking datasets
- train ranking NeuMF with `BCEWithLogitsLoss`
- evaluate ranking NeuMF with top-K metrics during training
- save trained model and encoders
- log final training to MLflow
- retain older rating-prediction training/evaluation utilities

### `src/explainability.py`

Explainability utilities.

Responsibilities:

- normalize hybrid weights
- calculate hybrid component contributions
- generate explanation sentences
- calculate exact SHAP-style values for structured hybrid features
- use SHAP when available
- calculate genre occlusion effects for neural scores

### `src/mlflow_utils.py`

Small helper module for MLflow.

Responsibilities:

- configure local MLflow tracking with SQLite
- set experiment names
- sanitize metric names
- log parameters
- log metrics
- log artifacts

### `src/content_engine.py`

Content-based recommendation engine.

Responsibilities:

- use movie metadata text features
- compute TF-IDF vectors
- compute cosine similarity
- recommend movies similar to a query title

This is useful as a content-based baseline and demo support.

### `src/hybrid_engine.py`

Older hybrid recommender module.

This file appears to be an earlier hybrid implementation. The final selected
hybrid system is `src/hybrid_ranker.py`.

### `notebooks/eda.ipynb`

Exploratory data analysis notebook.

It contains the original dataset analysis used to understand ratings, movies,
genres, sparsity, and modeling direction.

### `notebooks/final_submission.ipynb`

Final executable notebook.

It includes:

- artifact readiness checks
- dataset snapshot
- final benchmark visualization
- ranking NeuMF training history
- MLflow verification
- written interpretability section
- live explainable recommendation demo
- final project conclusion

This is the notebook intended to run cleanly from top to bottom for submission.

### `reports/topk_evaluation_summary.csv`

Latest top-K benchmark summary.

This is the main result table used in the README and final notebook.

### `reports/evaluation_runs/`

Timestamped evaluation history.

Contains:

- summary files
- per-user metrics
- metadata files
- leaderboard

This makes experiments traceable over time.

### `reports/neumf_ranking_training_history.csv`

Training history for the final ranking NeuMF model.

Contains:

- epoch
- training loss
- Precision@10
- Recall@10
- HitRate@10
- NDCG@10
- best-epoch flag

### `reports/hybrid_weight_tuning.csv`

Results from hybrid weight search.

Used to choose a good balance between:

- item-item KNN
- popularity
- genre affinity
- diversity penalty

### `reports/final_model_summary.md`

Concise final written summary of the selected model, evaluation protocol,
results, MLflow logging, and explainability approach.

### `models/`

Stores trained model artifacts.

Important artifacts:

```text
models/neumf_ranking_model.pth
models/neumf_model_small.pth
models/neumf_model_1M.pth
models/ranking_user_encoder.pkl
models/ranking_item_encoder.pkl
models/ranking_genre_cols.pkl
```

The ranking model artifacts are needed to evaluate or reuse the final ranking
NeuMF.

## Important Commands

Prepare data:

```powershell
python main.py --mode data
```

Train ranking NeuMF:

```powershell
python main.py --mode train_rank_neumf --ranking-epochs 5 --negative-samples 4
```

Run final sampled top-K benchmark:

```powershell
python main.py --mode eval_topk --run-label benchmark
```

Run full-catalog smoke benchmark:

```powershell
python main.py --mode eval_topk --full-catalog --max-users 25 --run-label full_catalog_smoke
```

Tune hybrid weights:

```powershell
python main.py --mode tune_hybrid --weight-step 0.25 --diversity-values 0 0.05 0.1
```

Run Streamlit app:

```powershell
python -m streamlit run app.py
```

Open MLflow:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run final notebook:

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/final_submission.ipynb --inplace --ExecutePreprocessor.timeout=600
```

## Final Model Decision

The final application uses the hybrid ranker.

Reason:

- best final sampled `NDCG@10`
- strong `HitRate@10`
- clear explanation structure
- fast enough for interactive use
- combines collaborative, popularity, and content signals

The ranking NeuMF remains important because it proves that the neural approach
works when trained with the correct ranking objective. However, for the final
application, the hybrid ranker gives the best balance of quality,
interpretability, and demo readiness.

## Limitations

The project is strong enough for submission, but there are still realistic
limitations:

- sampled negative evaluation is faster but less strict than full-catalog ranking
- MovieLens metadata is limited mostly to genres and titles
- the app does not yet include poster images
- ranking NeuMF explanations are less direct than hybrid explanations
- cold-start users and brand-new movies are not fully solved
- diversity is genre-based, so it does not capture all forms of recommendation
  variety

## Possible Future Improvements

Good next improvements would be:

- evaluate on a larger full-catalog sample
- add poster images through TMDB or IMDb links
- add a ranking NeuMF mode to the app for comparison
- improve content features with tags, plot summaries, or text embeddings
- add cold-start onboarding where a user selects favorite genres or movies
- add more advanced diversity methods
- add model cards or explanation quality tests

## Final Summary

The project evolved from a basic recommender into a complete explainable
recommendation system. The most important technical improvement was changing the
focus from rating prediction to top-K ranking quality. The strongest final model
for the app is the hybrid ranker, while the strongest neural model is the
ranking NeuMF.

The final system is not just a model. It includes data preparation, model
training, ranking evaluation, MLflow tracking, explainability, a final notebook,
and a working Streamlit GUI.
