import argparse # command-line parsing

def run_pipeline(args):
    """
    Orchestrates the sequential execution of the RecSys components.
    The pipeline is modular, allowing for full execution or targeted runs of specific steps.
    """
    mode = args.mode
    print("========================================")
    print("STARTING RECSYS PIPELINE")
    print("========================================\n")

    # Step 1: Data Preparation
    # Consolidates raw CSV files into an optimized Parquet format
    if mode in ['all', 'data']:
        from src.data_pipeline import prep_master_data

        print(">>> STEP 1: PREPPING DATA SYSTEM...")
        prep_master_data()
    
    # Step 2: Model Training (Collaborative Filtering)
    # Executes the NeuMF training loop using a high-volume data pipeline
    if mode in ['all', 'train_cf']:
        from src.neural_cf import train_hybrid_model

        print("\n>>> STEP 2: TRAINING NEURAL CF (Pipeline A)...")
        train_hybrid_model(epochs=10, batch_size=256)

    # Step 2b: Ranking NeuMF Training
    # Optimizes top-K relevance with sampled negatives and BCE logits
    if mode in ['train_rank_neumf']:
        from src.neural_cf import train_neumf_ranking_model

        print("\n>>> STEP 2b: TRAINING RANKING NEUMF...")
        train_neumf_ranking_model(
            epochs=args.ranking_epochs,
            batch_size=args.ranking_batch_size,
            negatives_per_positive=args.negative_samples,
            min_positive_rating=args.min_positive_rating,
            learning_rate=args.ranking_lr,
            validation_k=args.tune_k,
            validation_users=args.max_users,
            validation_negatives=args.negatives_per_user,
            random_state=args.random_state,
        )

    # Step 3: Performance Validation
    # Computes RMSE/MAE on the temporal test split to assess rating accuracy
    if mode in ['all', 'eval_cf']:
        from src.neural_cf import evaluate_cf_model

        print("\n>>> STEP 3: EVALUATING NEURAL CF...")
        evaluate_cf_model(batch_size=256)

    # Step 3b: Top-K Recommendation Evaluation
    # Measures ranking quality, not just star-rating prediction accuracy
    if mode in ['all', 'eval_topk']:
        from src.evaluation import evaluate_topk_recommenders

        print("\n>>> STEP 3b: EVALUATING TOP-K RECOMMENDATION QUALITY...")
        negatives_per_user = None if args.full_catalog else args.negatives_per_user
        evaluate_topk_recommenders(
            k_values=tuple(args.k_values),
            negatives_per_user=negatives_per_user,
            max_users=args.max_users,
            min_positive_rating=args.min_positive_rating,
            random_state=args.random_state,
            include_neumf=not args.skip_neumf,
            include_hybrid=not args.skip_hybrid,
            diversity_weight=args.diversity_weight,
            run_label=args.run_label,
            save_history=not args.no_history,
        )

    # Step 3c: Hybrid Weight Tuning
    # Searches weighted hybrid configurations against NDCG@K
    if mode in ['tune_hybrid']:
        from src.evaluation import tune_hybrid_weights

        print("\n>>> STEP 3c: TUNING HYBRID RANKER WEIGHTS...")
        negatives_per_user = None if args.full_catalog else args.negatives_per_user
        tune_hybrid_weights(
            k=args.tune_k,
            weight_step=args.weight_step,
            diversity_values=tuple(args.diversity_values),
            min_positive_rating=args.min_positive_rating,
            negatives_per_user=negatives_per_user,
            max_users=args.max_users,
            random_state=args.random_state,
        )

    # Step 4: Content Engine Testing
    # Validates the TF-IDF and Cosine Similarity logic using a sample query
    if mode in ['all', 'test_content']:
        from src.content_engine import get_content_recommendations

        print("\n>>> STEP 4: TESTING CONTENT ENGINE (Pipeline B)...")
        recs = get_content_recommendations("Matrix, The", top_n=5)
        print("\nTop Matches for 'The Matrix':")
        print(recs)

    print("\n========================================")
    print("PIPELINE EXECUTION COMPLETE")
    print("========================================")

if __name__ == "__main__":
    # Command-Line Argument Parsing
    # Provides a user-friendly interface to control the modular execution flow
    parser = argparse.ArgumentParser(description="Run the Recommendation System Pipeline")
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all', 
        choices=['all', 'data', 'train_cf', 'train_rank_neumf', 'eval_cf', 'eval_topk', 'tune_hybrid', 'test_content'],
        help="Which part of the pipeline to run."
    )
    parser.add_argument(
        '--full-catalog',
        action='store_true',
        help="Evaluate against all unseen movies instead of sampled negatives."
    )
    parser.add_argument(
        '--negatives-per-user',
        type=int,
        default=100,
        help="Number of unrated negative candidates to sample per user for eval_topk."
    )
    parser.add_argument(
        '--max-users',
        type=int,
        default=100,
        help="Maximum number of users to evaluate for eval_topk. Use 0 for all users."
    )
    parser.add_argument(
        '--min-positive-rating',
        type=float,
        default=4.0,
        help="Held-out ratings at or above this value count as positives."
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[5, 10],
        help="K values for Precision@K, Recall@K, HitRate@K, and NDCG@K."
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help="Random seed for user and negative sampling."
    )
    parser.add_argument(
        '--skip-neumf',
        action='store_true',
        help="Skip NeuMF scoring during eval_topk."
    )
    parser.add_argument(
        '--skip-hybrid',
        action='store_true',
        help="Skip hybrid ranker scoring during eval_topk."
    )
    parser.add_argument(
        '--diversity-weight',
        type=float,
        default=0.0,
        help="Diversity penalty strength for the hybrid ranker during eval_topk."
    )
    parser.add_argument(
        '--weight-step',
        type=float,
        default=0.25,
        help="Grid step for tune_hybrid weight search."
    )
    parser.add_argument(
        '--diversity-values',
        type=float,
        nargs='+',
        default=[0.0, 0.05, 0.10],
        help="Diversity penalty values to test during tune_hybrid."
    )
    parser.add_argument(
        '--tune-k',
        type=int,
        default=10,
        help="K value optimized by tune_hybrid."
    )
    parser.add_argument(
        '--ranking-epochs',
        type=int,
        default=5,
        help="Epochs for train_rank_neumf."
    )
    parser.add_argument(
        '--ranking-batch-size',
        type=int,
        default=1024,
        help="Batch size for train_rank_neumf."
    )
    parser.add_argument(
        '--negative-samples',
        type=int,
        default=4,
        help="Number of sampled negative interactions per positive for train_rank_neumf."
    )
    parser.add_argument(
        '--ranking-lr',
        type=float,
        default=0.001,
        help="Learning rate for train_rank_neumf."
    )
    parser.add_argument(
        '--run-label',
        type=str,
        default=None,
        help="Optional label included in timestamped evaluation run files."
    )
    parser.add_argument(
        '--no-history',
        action='store_true',
        help="Only write reports/topk_evaluation_summary.csv; do not append run history."
    )
    
    args = parser.parse_args()
    if args.max_users == 0:
        args.max_users = None
    run_pipeline(args)
