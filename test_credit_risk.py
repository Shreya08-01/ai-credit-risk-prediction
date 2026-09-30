import pytest
import numpy as np
import pandas as pd
from data.generate_dataset import generate_synthetic_credit_data
from src.data_loader import clean_data, prepare_train_test_data
from src.preprocessing import CreditFeatureEngineer, CreditDataPipeline
from src.scorecard import CreditScorecardEngine
from src.models import (
    CreditRiskModelTrainer, calculate_gini_coefficient, calculate_ks_statistic,
    roc_auc_score_numpy, LogisticRegressionNumPy, CreditRiskPipelineWrapper
)
from src.explainability import explain_applicant_risk, get_global_feature_importance

def test_dataset_generation():
    df = generate_synthetic_credit_data(num_samples=100)
    assert len(df) == 100
    assert 'loan_status' in df.columns
    assert set(df['loan_status'].unique()).issubset({0, 1})
    assert df['credit_score'].min() >= 300
    assert df['credit_score'].max() <= 850

def test_data_cleaning_and_split():
    df = generate_synthetic_credit_data(num_samples=100)
    df_clean = clean_data(df)
    assert df_clean.isnull().sum().sum() == 0
    
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_clean, test_size=0.25)
    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25

def test_feature_engineering_and_pipeline():
    df = generate_synthetic_credit_data(num_samples=50)
    fe = CreditFeatureEngineer()
    transformed_df = fe.transform(df)
    
    assert 'engineered_loan_to_income' in transformed_df.columns
    assert 'engineered_repayment_to_income' in transformed_df.columns
    assert 'engineered_cred_hist_to_age' in transformed_df.columns
    assert 'engineered_high_int_rate_flag' in transformed_df.columns
    
    pipeline = CreditDataPipeline()
    matrix = pipeline.fit_transform(df)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == 50
    assert matrix.shape[1] > 10

def test_scorecard_engine():
    engine = CreditScorecardEngine(target_base_score=600, pdo=50)
    
    high_risk_score = engine.probability_to_score(0.95)
    low_risk_score = engine.probability_to_score(0.02)
    
    assert 300 <= high_risk_score <= 850
    assert 300 <= low_risk_score <= 850
    assert low_risk_score > high_risk_score
    
    low_eval = engine.evaluate_applicant(0.01)
    high_eval = engine.evaluate_applicant(0.90)
    
    assert low_eval['recommendation'] == 'APPROVE'
    assert low_eval['risk_tier'] == 'Low Risk'
    assert high_eval['recommendation'] == 'DECLINE'
    assert high_eval['risk_tier'] == 'High Risk'

def test_gini_and_ks_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    auc = roc_auc_score_numpy(y_true, y_proba)
    assert auc == 1.0
    
    gini = calculate_gini_coefficient(auc)
    assert gini == 1.0
    
    ks = calculate_ks_statistic(y_true, y_proba)
    assert ks == 1.0

def test_model_wrapper_training():
    df = generate_synthetic_credit_data(num_samples=100)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    wrapper = CreditRiskPipelineWrapper("Logistic Regression", LogisticRegressionNumPy(n_iter=100))
    wrapper.fit(X_train, y_train)
    
    probas = wrapper.predict_proba(X_test)
    preds = wrapper.predict(X_test)
    
    assert probas.shape == (len(X_test), 2)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})
    
    imp_df = get_global_feature_importance(wrapper)
    assert len(imp_df) > 0
    assert 'importance' in imp_df.columns

def test_explainability_rules():
    applicant = pd.DataFrame([{
        'loan_percent_income': 0.45,
        'credit_score': 550,
        'cb_person_default_on_file': 'Y',
        'person_emp_length': 1,
        'loan_int_rate': 18.5,
        'person_home_ownership': 'RENT'
    }])
    
    exp = explain_applicant_risk(applicant)
    assert 'risk_drivers' in exp
    assert 'mitigating_factors' in exp
    assert len(exp['risk_drivers']) > 0
