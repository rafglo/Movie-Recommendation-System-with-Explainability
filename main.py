import argparse

# Import the functional logic from our src/ folder
from src.data_pipeline import prep_master_data
from src.neural_cf import train_hybrid_model, evaluate_cf_model
from src.content_engine import get_content_recommendations

def run_pipeline(mode):
    print("========================================")
    print("🎬 STARTING RECSYS PIPELINE 🎬")
    print("========================================\n")

    if mode in ['all', 'data']:
        print(">>> STEP 1: PREPPING DATA SYSTEM...")
        # UPDATED: Point to the new 1M row parameters
        prep_master_data(target_rows=1000000, min_ratings=20)
    
    if mode in ['all', 'train_cf']:
        print("\n>>> STEP 2: TRAINING NEURAL CF (Pipeline A)...")
        train_hybrid_model(epochs=10, batch_size=2048)

    if mode in ['all', 'eval_cf', 'train_cf']:
        print("\n>>> STEP 3: EVALUATING NEURAL CF...")
        # UPDATED: Massive batch size for evaluation
        evaluate_cf_model(batch_size=2048)

    if mode in ['all', 'test_content']:
        print("\n>>> STEP 4: TESTING CONTENT ENGINE (Pipeline B)...")
        recs = get_content_recommendations("Matrix, The", top_n=5)
        print("\nTop Matches for 'The Matrix':")
        print(recs)

    print("\n========================================")
    print("✅ PIPELINE EXECUTION COMPLETE ✅")
    print("========================================")

if __name__ == "__main__":
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