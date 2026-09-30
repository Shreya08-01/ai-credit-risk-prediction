import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """Loads credit dataset from CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Credit dataset not found at path: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans credit data by handling missing values and data types."""
    df_clean = df.copy()
    
    num_cols = df_clean.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean

def prepare_train_test_data(
    df: pd.DataFrame, target_col: str = 'loan_status', test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits credit dataframe into features and target train/test sets with stratified sampling."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in dataset.")
        
    np.random.seed(random_state)
    
    # Stratified index selection
    target_vals = df[target_col].values
    unique_classes = np.unique(target_vals)
    
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        cls_idx = np.where(target_vals == cls)[0]
        np.random.shuffle(cls_idx)
        n_test = int(len(cls_idx) * test_size)
        test_indices.extend(cls_idx[:n_test])
        train_indices.extend(cls_idx[n_test:])
        
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, y_train = X.iloc[train_indices].reset_index(drop=True), y.iloc[train_indices].reset_index(drop=True)
    X_test, y_test = X.iloc[test_indices].reset_index(drop=True), y.iloc[test_indices].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test
