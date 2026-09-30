import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

class CreditFeatureEngineer:
    """Domain-specific feature engineering transformer for credit data."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X_df = df.copy()
        
        income = np.maximum(X_df['person_income'].values.astype(float), 1.0)
        age = np.maximum(X_df['person_age'].values.astype(float), 18.0)
        loan_amnt = X_df['loan_amnt'].values.astype(float)
        int_rate = X_df['loan_int_rate'].values.astype(float)
        cred_hist = X_df['cb_person_cred_hist_length'].values.astype(float)
        
        X_df['engineered_loan_to_income'] = np.round(loan_amnt / income, 4)
        estimated_repayment = loan_amnt * (1.0 + int_rate / 100.0)
        X_df['engineered_repayment_to_income'] = np.round(estimated_repayment / income, 4)
        X_df['engineered_cred_hist_to_age'] = np.round(cred_hist / age, 4)
        X_df['engineered_high_int_rate_flag'] = (int_rate > 15.0).astype(int)
        
        return X_df

class CreditDataPipeline:
    """
    Robust NumPy/Pandas Data Pipeline for scaling, one-hot encoding, and feature transformation
    without external SciPy DLL dependencies.
    """
    
    def __init__(self):
        self.feature_engineer = CreditFeatureEngineer()
        self.numeric_cols = [
            'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
            'engineered_loan_to_income', 'engineered_repayment_to_income',
            'engineered_cred_hist_to_age', 'engineered_high_int_rate_flag'
        ]
        self.categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        
        self.means = {}
        self.stds = {}
        self.medians = {}
        self.cat_categories = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'CreditDataPipeline':
        df_engineered = self.feature_engineer.transform(df)
        
        # Fit numerical medians, means, stds
        for col in self.numeric_cols:
            median_val = df_engineered[col].median()
            self.medians[col] = median_val
            filled_col = df_engineered[col].fillna(median_val).values
            self.means[col] = np.mean(filled_col)
            std_val = np.std(filled_col)
            self.stds[col] = std_val if std_val > 1e-6 else 1.0

        # Fit categorical categories
        for col in self.categorical_cols:
            cats = sorted(df_engineered[col].dropna().unique().tolist())
            self.cat_categories[col] = cats

        # Build feature names
        feature_names = list(self.numeric_cols)
        for col in self.categorical_cols:
            for cat in self.cat_categories[col]:
                feature_names.append(f"{col}_{cat}")
                
        self.feature_names = feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
            
        df_engineered = self.feature_engineer.transform(df)
        num_samples = len(df_engineered)
        
        # Process numeric features
        num_matrix = np.zeros((num_samples, len(self.numeric_cols)))
        for i, col in enumerate(self.numeric_cols):
            vals = df_engineered[col].fillna(self.medians[col]).values.astype(float)
            num_matrix[:, i] = (vals - self.means[col]) / self.stds[col]
            
        # Process categorical features (One-Hot Encoding)
        cat_matrices = []
        for col in self.categorical_cols:
            cats = self.cat_categories[col]
            col_vals = df_engineered[col].astype(str).values
            one_hot = np.zeros((num_samples, len(cats)))
            for j, cat in enumerate(cats):
                one_hot[:, j] = (col_vals == str(cat)).astype(float)
            cat_matrices.append(one_hot)
            
        if cat_matrices:
            cat_combined = np.hstack(cat_matrices)
            return np.hstack([num_matrix, cat_combined])
        return num_matrix

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
