"""
Preprocessing Pipeline Module for Internet Usage Forecasting (Streamlined & Relevant Features).
Focuses strictly on the most relevant features:
- Core digital activities: video streaming, downloads, video calls, gaming, social media
- Key hardware footprint: smart TV, laptop, mobile, connected devices
- Essential temporal & lag dynamics: previous_day_usage, rolling_7_day_average, is_weekend
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.feature_engineering import create_lag_and_rolling_features, backfill_initial_lags

FEATURE_METADATA_PATH = os.path.join("models", "feature_columns.json")
PIPELINE_PATH = os.path.join("models", "preprocessing_pipeline.pkl")

# Only the few, most relevant numerical features (11 features)
NUMERICAL_FEATURES = [
    "video_streaming_hours",     # Strongest activity bandwidth driver
    "downloading_gb",            # Large direct volume additions
    "video_calling_hours",       # Work/school synchronous video stream
    "online_gaming_hours",       # High packet rate gaming
    "social_media_hours",        # Casual media browsing
    "smart_tv_usage_gb",         # Primary household video consumption screen
    "laptop_usage_gb",           # Computing & productivity device
    "mobile_usage_gb",           # Smartphone traffic
    "connected_devices_count",   # Total hardware footprint
    "previous_day_usage",        # 1-day auto-regressive momentum
    "rolling_7_day_average"      # 7-day baseline trend
]

# Essential temporal indicator (1 feature)
CATEGORICAL_FEATURES = [
    "is_weekend"                 # Weekend leisure vs weekday routine
]

TARGET_COL = "total_data_usage_gb"

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity cleaning, date conversion, and deduplication."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df

def build_preprocessing_pipeline(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Construct a clean scikit-learn ColumnTransformer.
    - Numerical: SimpleImputer(median) -> StandardScaler
    - Categorical/Binary: SimpleImputer(most_frequent) (kept numeric 0/1, no expansion)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor

def prepare_dataset_for_modeling(
    df: pd.DataFrame,
    test_size: float = 0.2,
    save_artifacts: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Full streamlined preprocessing flow:
    1. Clean raw data and ensure chronological ordering.
    2. Engineer only essential lags (previous day & rolling 7-day).
    3. Chronologically split into train and test sets (no shuffling!).
    4. Fit ColumnTransformer on train only, transform train and test.
    5. Save pipeline and feature names.
    """
    df_cleaned = clean_raw_data(df)
    df_featured = create_lag_and_rolling_features(df_cleaned, target_col=TARGET_COL)
    df_featured = backfill_initial_lags(df_featured, target_col=TARGET_COL)
    
    # Chronological Split
    total_len = len(df_featured)
    split_idx = int(total_len * (1 - test_size))
    
    train_df = df_featured.iloc[:split_idx].copy()
    test_df = df_featured.iloc[split_idx:].copy()
    
    # Available features in data
    num_cols = [c for c in NUMERICAL_FEATURES if c in df_featured.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df_featured.columns]
    
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    
    # Fit strictly on train
    X_train_raw = train_df[num_cols + cat_cols]
    y_train = train_df[TARGET_COL].values
    
    X_test_raw = test_df[num_cols + cat_cols]
    y_test = test_df[TARGET_COL].values
    
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    
    all_feature_names = num_cols + cat_cols
    
    if save_artifacts:
        os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
        joblib.dump(preprocessor, PIPELINE_PATH)
        
        # Save feature columns metadata as JSON
        import json
        metadata = {
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "all_feature_names": all_feature_names,
            "total_feature_count": len(all_feature_names),
            "target": TARGET_COL,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_date_range": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
            "test_date_range": [str(test_df["date"].min().date()), str(test_df["date"].max().date())]
        }
        with open(FEATURE_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
            
    return X_train_trans, X_test_trans, y_train, y_test, preprocessor, all_feature_names, train_df, test_df
