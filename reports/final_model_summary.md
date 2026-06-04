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

Full-catalog evaluation is also available with:

```powershell
python main.py --mode eval_topk --full-catalog
```

## Final Sampled Results

| Recommender | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| ranking NeuMF | 0.416 | 0.498 | 0.930 | 0.586 |
| hybrid ranker | 0.397 | 0.504 | 0.950 | 0.584 |
| item-item KNN | 0.382 | 0.483 | 0.940 | 0.541 |
| popularity baseline | 0.330 | 0.381 | 0.850 | 0.473 |
| rating NeuMF | 0.195 | 0.210 | 0.760 | 0.241 |
| random baseline | 0.104 | 0.120 | 0.620 | 0.136 |

## Selected GUI Model

The Streamlit app uses the hybrid ranker.

Reason:

- it is nearly tied with ranking NeuMF on NDCG@10
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
| 1 | 0.572 | 0.221 |
| 2 | 0.458 | 0.398 |
| 3 | 0.416 | 0.511 |
| 4 | 0.372 | 0.558 |
| 5 | 0.333 | 0.593 |

## Explainability Strategy

Model-specific explanations are used first:

- item-item KNN: recommended because it is similar to movies the user liked
- hybrid ranker: exact weighted component contributions
- ranking NeuMF: genre occlusion sensitivity

SHAP is used only on final structured hybrid features:

- item KNN score
- popularity score
- genre score

This avoids explaining opaque embedding dimensions directly.

## Next Recommended Work

- add screenshots to the final presentation or report
- optionally add a ranking NeuMF app mode
- evaluate all models on a larger full-catalog user sample
- add poster images through `links.csv`/TMDB or another image source
- clean old exploratory run artifacts before final submission
