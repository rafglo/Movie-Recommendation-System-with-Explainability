import argparse
import os
import mlflow
import mlflow.pytorch

# Importy Twoich modułów
from src.data_pipeline import prep_master_data
from src.neural_cf import train_hybrid_model, evaluate_cf_model
from src.content_engine import get_content_recommendations

def run_pipeline(mode):
    """
    Orkiestrator potoku RecSys z integracją MLflow.
    """
    # Ustawienie nazwy eksperymentu w MLflow
    mlflow.set_experiment("Movie_Recommendation_System")

    print("========================================")
    print("STARTING RECSYS PIPELINE WITH MLFLOW")
    print("========================================\n")

    # Startujemy sesję śledzenia
    with mlflow.start_run(run_name=f"Run_{mode}"):
        
        # Logujemy tryb uruchomienia jako parametr
        mlflow.log_param("pipeline_mode", mode)
# Krok 1: Przygotowanie danych
        if mode in ['all', 'data']:
            print(">>> STEP 1: PREPPING DATA SYSTEM...")
            prep_master_data()
            # POPRAWKA TUTAJ:
            mlflow.set_tag("step_1", "data_ready")

        # Krok 2: Trening Modelu
        model = None
        if mode in ['all', 'train_cf']:
            epochs = 10
            batch_size = 256
            print(f"\n>>> STEP 2: TRAINING NEURAL CF (Epochs: {epochs})...")
            
            # Logujemy parametry treningu
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "model_type": "NeuMF"
            })
            
            # Zakładamy, że funkcja zwraca model: return model
            model = train_hybrid_model(epochs=epochs, batch_size=batch_size)
            
            # Logujemy sam model do MLflow
            if model is not None:
                mlflow.pytorch.log_model(model, "trained_model")

        # Krok 3: Ewaluacja
        if mode in ['all', 'eval_cf']:
            print("\n>>> STEP 3: EVALUATING NEURAL CF...")
            # Zakładamy, że funkcja zwraca wyniki: return rmse, mae
            rmse, mae = evaluate_cf_model(batch_size=256)
            
            # LOGOWANIE METRYK DO MLFLOW
            mlflow.log_metrics({
                "test_rmse": rmse,
                "test_mae": mae
            })
            print(f"Metrics logged: RMSE={rmse:.4f}, MAE={mae:.4f}")

        # Krok 4: Testowanie silnika treściowego
        if mode in ['all', 'test_content']:
            print("\n>>> STEP 4: TESTING CONTENT ENGINE...")
            movie_query = "Matrix, The"
            recs = get_content_recommendations(movie_query, top_n=5)
            print(f"\nTop Matches for '{movie_query}':")
            print(recs)
            # Możemy zalogować wynik jako tekstowy artefakt
            mlflow.log_text(str(recs), "content_recommendations_sample.txt")

    print("\n========================================")
    print("PIPELINE EXECUTION COMPLETE & LOGGED")
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