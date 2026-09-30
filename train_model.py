import joblib
import pandas as pd
from pathlib import Path
from data.generate_dataset import generate_synthetic_credit_data
from src.data_loader import clean_data, prepare_train_test_data
from src.models import CreditRiskModelTrainer
from src.scorecard import CreditScorecardEngine

def main():
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_dir / "credit_risk_dataset.csv"
    
    # 1. Dataset Generation or Loading
    if not dataset_path.exists():
        print("[+] Generating synthetic credit risk dataset (5,000 samples)...")
        df = generate_synthetic_credit_data(num_samples=5000)
        df.to_csv(dataset_path, index=False)
    else:
        print(f"[+] Loading existing credit risk dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
    df = clean_data(df)
    
    # 2. Train / Test Split
    print("[+] Splitting data into 80% train / 20% test sets...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(df, target_col='loan_status')
    
    # 3. Model Training & Benchmarking
    print("[+] Training Logistic Regression, Random Forest, and XGBoost/GBDT models...")
    trainer = CreditRiskModelTrainer(random_state=42)
    summary = trainer.train_all(X_train, y_train, X_test, y_test)
    
    # 4. Display Results Summary Table
    print("\n" + "=" * 70)
    print(f"{'MODEL PERFORMANCE BENCHMARK SUMMARY':^70}")
    print("=" * 70)
    print(f"{'Model Name':<20} | {'ROC-AUC':<9} | {'Gini':<9} | {'KS Stat':<9} | {'F1 Score':<9}")
    print("-" * 70)
    
    for name, metrics in summary['evaluations'].items():
        print(
            f"{name:<20} | "
            f"{metrics['roc_auc']:<9.4f} | "
            f"{metrics['gini']:<9.4f} | "
            f"{metrics['ks_statistic']:<9.4f} | "
            f"{metrics['f1_score']:<9.4f}"
        )
    print("=" * 70)
    print(f"[*] Best Model Selected: {summary['best_model_name']}")
    print("=" * 70 + "\n")
    
    # 5. Save Artifacts
    best_wrapper = trainer.best_wrapper
    scorecard_engine = CreditScorecardEngine()
    
    artifact = {
        'model_name': summary['best_model_name'],
        'wrapper': best_wrapper,
        'scorecard_engine': scorecard_engine,
        'evaluations': summary['evaluations']
    }
    
    artifact_path = models_dir / "credit_risk_model.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"[+] Best credit risk model saved to: {artifact_path}")

if __name__ == "__main__":
    main()
