"""
Feature Engineering Module for Internet Usage Forecasting (Streamlined & Relevant).
Calculates essential historical lag (previous_day_usage) and rolling trend (rolling_7_day_average)
strictly avoiding lookahead data leakage.
"""

import numpy as np
import pandas as pd

def create_lag_and_rolling_features(df: pd.DataFrame, target_col: str = "total_data_usage_gb") -> pd.DataFrame:
    """
    Construct essential time-series lag and rolling statistics on the target variable.
    All rolling windows are shifted by 1 day so they strictly reflect past consumption
    and do not leak the current day's target.
    """
    df = df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 1. Immediate previous day usage (Auto-regressive lag 1)
    df["previous_day_usage"] = df[target_col].shift(1)
    
    # 2. Shifted 7-day rolling baseline (Weekly trend without lookahead)
    past_series = df[target_col].shift(1)
    df["rolling_7_day_average"] = past_series.rolling(window=7, min_periods=1).mean()
    
    return df

def backfill_initial_lags(df: pd.DataFrame, target_col: str = "total_data_usage_gb") -> pd.DataFrame:
    """
    Backfill the initial unobserved lag days using series median
    to prevent row loss in early time-series periods.
    """
    df = df.copy()
    initial_median = df[target_col].median()
    
    for col in ["previous_day_usage", "rolling_7_day_average"]:
        if col in df.columns:
            df[col] = df[col].bfill().fillna(initial_median)
            
    return df
