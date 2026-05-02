import argparse # command-line parsing

# functional logic from our src/ folder
from src.data_pipeline import prep_master_data
from src.neural_cf import train_hybrid_model, evaluate_cf_model
from src.content_engine import get_content_recommendations

def run_pipeline(mode):
    """
    Orchestrates the sequential execution of the RecSys components.
    The pipeline is modular, allowing for full execution or targeted runs of specific steps.
    """
    print("========================================")
    print("STARTING RECSYS PIPELINE")
    print("========================================\n")

    # Step 1: Data Preparation
    # Consolidates raw CSV files into an optimized Parquet format
    if mode in ['all', 'data']:
        print(">>> STEP 1: PREPPING DATA SYSTEM...")
        prep_master_data()
    
    # Step 2: Model Training (Collaborative Filtering)
    # Executes the NeuMF training loop using a high-volume data pipeline
    if mode in ['all', 'train_cf']:
        print("\n>>> STEP 2: TRAINING NEURAL CF (Pipeline A)...")
        train_hybrid_model(epochs=10, batch_size=256)

    # Step 3: Performance Validation
    # Computes RMSE/MAE on the temporal test split to assess rating accuracy
    if mode in ['all', 'eval_cf']:
        print("\n>>> STEP 3: EVALUATING NEURAL CF...")
        evaluate_cf_model(batch_size=256)

    # Step 4: Content Engine Testing
    # Validates the TF-IDF and Cosine Similarity logic using a sample query
    if mode in ['all', 'test_content']:
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
        choices=['all', 'data', 'train_cf', 'eval_cf', 'test_content'],
        help="Which part of the pipeline to run."
    )
    
    args = parser.parse_args()
    run_pipeline(args.mode)