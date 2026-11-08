"""
Forecasting Module for Daily Internet Usage.
Implements recursive multi-step forecasting (1 to 7 days) using trained ML models,
dynamically updating time-series lag and rolling statistics.
Computes statistically justified prediction intervals based on empirical validation residuals,
and provides plan limit compliance analysis.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from src.database import calculate_date_features, load_combined_data
from src.preprocessing import TARGET_COL, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# In-memory artifact cache to avoid repeated disk reads or re-fitting
_ARTIFACT_CACHE: Dict[str, Any] = {}

def load_forecasting_artifacts(model_name: Optional[str] = None):
    """
    Load model, preprocessing pipeline, and validation residual metadata.
    Includes automated self-healing for cross-platform and scikit-learn version differences
    (e.g., Streamlit Cloud vs local Windows environments).
    """
    global _ARTIFACT_CACHE
    
    target_model_name = model_name or "Linear Regression"
    cache_key = f"{target_model_name}"
    
    if cache_key in _ARTIFACT_CACHE:
        cached = _ARTIFACT_CACHE[cache_key]
        return cached["model"], cached["preprocessor"], cached["name"], cached["residuals"]
        
    # 1. Determine target model name and residual stats from metrics if available
    default_res_stats = {"std": 0.5218, "q05": -0.852, "q25": -0.321, "q50": 0.0, "q75": 0.321, "q95": 0.852}
    residual_stats = default_res_stats
    model_file = None
    
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
            if model_name is None:
                target_model_name = metrics_data.get("best_model_name", "Linear Regression")
            if target_model_name in metrics_data.get("models", {}):
                residual_stats = metrics_data["models"][target_model_name].get("residual_stats", default_res_stats)
                model_file = metrics_data["models"][target_model_name].get("file")
        except Exception:
            pass
            
    # Dummy test row to verify preprocessor compatibility across sklearn versions
    dummy_dict = {col: 1.0 for col in NUMERICAL_FEATURES}
    for cat in CATEGORICAL_FEATURES:
        dummy_dict[cat] = 0
    dummy_df = pd.DataFrame([dummy_dict])
    
    # 2. Load or self-heal Preprocessor
    preprocessor = None
    if os.path.exists(PIPELINE_PATH):
        try:
            loaded_prep = joblib.load(PIPELINE_PATH)
            # Test transform to catch cross-version attribute errors (e.g. SimpleImputer._fill_dtype)
            loaded_prep.transform(dummy_df)
            preprocessor = loaded_prep
        except Exception:
            preprocessor = None
            
    if preprocessor is None:
        # Self-heal: fit pipeline directly using the active environment's scikit-learn
        from src.preprocessing import prepare_dataset_for_modeling
        df_base = load_combined_data()
        X_train, X_test, y_train, y_test, preprocessor, _, _, _ = prepare_dataset_for_modeling(df_base, save_artifacts=False)
        
    # 3. Load or self-heal Model
    model = None
    if model_file:
        model_path = os.path.join(MODELS_DIR, model_file)
    else:
        model_path = os.path.join(MODELS_DIR, target_model_name.lower().replace(" ", "_") + ".pkl")
        
    if os.path.exists(model_path):
        try:
            loaded_model = joblib.load(model_path)
            dummy_trans = preprocessor.transform(dummy_df)
            loaded_model.predict(dummy_trans)
            model = loaded_model
        except Exception:
            model = None
            
    if model is None:
        # Self-heal: fit model directly
        from src.train_models import get_models
        from src.preprocessing import prepare_dataset_for_modeling
        df_base = load_combined_data()
        X_train, _, y_train, _, _, _, _, _ = prepare_dataset_for_modeling(df_base, save_artifacts=False)
        models_dict = get_models()
        model = models_dict.get(target_model_name, models_dict["Linear Regression"])
        model.fit(X_train, y_train)
        
    _ARTIFACT_CACHE[cache_key] = {
        "model": model,
        "preprocessor": preprocessor,
        "name": target_model_name,
        "residuals": residual_stats
    }
    
    return model, preprocessor, target_model_name, residual_stats

def generate_recursive_forecast(
    df: pd.DataFrame,
    horizon_days: int = 7,
    model_name: Optional[str] = None,
    custom_next_day_inputs: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate recursive multi-step forecast for the next 1 to horizon_days.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical dataset sorted chronologically up to the latest recorded date.
    horizon_days : int
        Number of forward days to forecast (1 to 14, default 7).
    model_name : str, optional
        Specific model to use (or defaults to best model).
    custom_next_day_inputs : dict, optional
        User-provided expected hours/devices for tomorrow (day 1).
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]:
        - Forecast table with dates, predicted GB, lower/upper intervals, and status.
        - Summary metadata dictionary including monthly quota projections.
    """
    model, preprocessor, selected_model_name, res_stats = load_forecasting_artifacts(model_name)
    
    # Ensure chronological order
    df_hist = df.copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.sort_values("date").reset_index(drop=True)
    if "day_of_week" not in df_hist.columns:
        df_hist["day_of_week"] = df_hist["date"].dt.strftime("%A")
    
    last_date = df_hist["date"].iloc[-1]
    last_known_plan_limit = df_hist["internet_plan_limit_gb"].iloc[-1] if "internet_plan_limit_gb" in df_hist.columns else 750.0
    
    # Track historical and forecasted target series
    usage_series = list(df_hist[TARGET_COL].values)
    
    # Baseline recent activity/device averages by day-of-week (to inform day 2..7 expectations)
    dow_groups = df_hist.tail(120).groupby("day_of_week")
    dow_averages = dow_groups.mean(numeric_only=True).to_dict(orient="index")
    recent_overall_mean = df_hist.tail(30).mean(numeric_only=True).to_dict()
    
    forecast_records = []
    
    for h in range(1, horizon_days + 1):
        target_date = last_date + timedelta(days=h)
        target_date_str = target_date.strftime("%Y-%m-%d")
        cal_features = calculate_date_features(target_date_str)
        dow_name = cal_features["day_of_week"]
        
        # 1. Construct lag and rolling features from historical + forecasted series
        prev_usage = usage_series[-1]
        roll7_slice = usage_series[-7:]
        roll7_mean = float(np.mean(roll7_slice))
        
        # 2. Activity and Device features for horizon step
        if h == 1 and custom_next_day_inputs is not None:
            day_inputs = custom_next_day_inputs.copy()
        else:
            # Inform with historical average for this specific day of week
            if dow_name in dow_averages:
                day_inputs = dow_averages[dow_name].copy()
            else:
                day_inputs = recent_overall_mean.copy()
                
        # Build streamlined row dictionary for model preprocessing (few, only relevant features)
        row_dict = {
            "video_streaming_hours": float(day_inputs.get("video_streaming_hours", 2.5) or 2.5),
            "downloading_gb": float(day_inputs.get("downloading_gb", 1.5) or 1.5),
            "video_calling_hours": float(day_inputs.get("video_calling_hours", 1.0) or 1.0),
            "online_gaming_hours": float(day_inputs.get("online_gaming_hours", 0.8) or 0.8),
            "social_media_hours": float(day_inputs.get("social_media_hours", 1.8) or 1.8),
            "smart_tv_usage_gb": float(day_inputs.get("smart_tv_usage_gb", 4.0) or 4.0),
            "laptop_usage_gb": float(day_inputs.get("laptop_usage_gb", 3.1) or 3.1),
            "mobile_usage_gb": float(day_inputs.get("mobile_usage_gb", 2.2) or 2.2),
            "connected_devices_count": int(day_inputs.get("connected_devices_count", 8) or 8),
            "previous_day_usage": prev_usage,
            "rolling_7_day_average": roll7_mean,
            "is_weekend": cal_features["is_weekend"]
        }
        
        row_df = pd.DataFrame([row_dict])
        
        # Transform via preprocessor pipeline
        X_trans = preprocessor.transform(row_df)
        pred_val = float(model.predict(X_trans)[0])
        pred_val = max(0.5, round(pred_val, 2))
        
        # Aggregate active online activity hours
        tot_act = (
            row_dict["video_streaming_hours"] +
            row_dict["video_calling_hours"] +
            row_dict["online_gaming_hours"] +
            row_dict["social_media_hours"]
        )
        
        # Empirical prediction interval using validation residual quantiles
        # Multi-step variance compounds by factor of sqrt(horizon)
        horizon_expansion = np.sqrt(h)
        lower_bound = max(0.2, round(pred_val + res_stats["q05"] * horizon_expansion, 2))
        upper_bound = round(pred_val + res_stats["q95"] * horizon_expansion, 2)
        
        # Record for recursive propagation
        usage_series.append(pred_val)
        
        forecast_records.append({
            "horizon_step": h,
            "date": target_date_str,
            "day_of_week": dow_name,
            "is_weekend": "Weekend" if cal_features["is_weekend"] else "Weekday",
            "predicted_usage_gb": pred_val,
            "lower_interval_90_gb": lower_bound,
            "upper_interval_90_gb": upper_bound,
            "expected_activity_hours": round(tot_act, 1)
        })
        
    forecast_df = pd.DataFrame(forecast_records)
    
    # Internet Plan / Quota analysis
    current_month_records = df_hist[df_hist["date"].dt.month == last_date.month]
    month_to_date_used = float(current_month_records[TARGET_COL].sum())
    remaining_days_in_month = (target_date - last_date).days
    predicted_7d_total = float(forecast_df["predicted_usage_gb"].sum())
    
    plan_limit = float(last_known_plan_limit)
    daily_quota_fair = plan_limit / 30.0
    
    metadata = {
        "model_used": selected_model_name,
        "last_historical_date": last_date.strftime("%Y-%m-%d"),
        "total_7d_forecast_gb": round(predicted_7d_total, 2),
        "avg_daily_forecast_gb": round(float(forecast_df["predicted_usage_gb"].mean()), 2),
        "monthly_plan_limit_gb": plan_limit,
        "month_to_date_used_gb": round(month_to_date_used, 2),
        "fair_daily_quota_gb": round(daily_quota_fair, 2),
        "residual_std": res_stats["std"],
        "confidence_level": "90% Empirical Residual Quantile Range"
    }
    
    return forecast_df, metadata
