"""
Anomaly Detection Module for Daily Internet Usage.
Implements Isolation Forest and IQR/Z-score based anomaly detection.
Diagnoses root causes (heavy downloads, streaming spikes, concurrent devices)
and distinguishes statistical high-usage from network security incidents.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.ensemble import IsolationForest

def detect_anomalies_iqr(
    df: pd.DataFrame,
    target_col: str = "total_data_usage_gb",
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect statistical outliers using the Interquartile Range (IQR) method.
    Upper threshold = Q3 + multiplier * IQR.
    """
    df = df.copy()
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    upper_threshold = q3 + (multiplier * iqr)
    lower_threshold = max(0.0, q1 - (multiplier * iqr))
    
    df["is_iqr_anomaly"] = (df[target_col] > upper_threshold) | (df[target_col] < lower_threshold)
    df["iqr_upper_threshold"] = upper_threshold
    return df

def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Detect multi-dimensional usage anomalies using Isolation Forest.
    Features: total_data_usage_gb, downloading_gb, video_streaming_hours, connected_devices_count.
    """
    df = df.copy()
    feature_candidates = [
        "total_data_usage_gb",
        "downloading_gb",
        "video_streaming_hours",
        "connected_devices_count"
    ]
    avail_features = [c for c in feature_candidates if c in df.columns]
    
    # Fill NAs with median for isolation forest fit
    X = df[avail_features].fillna(df[avail_features].median())
    
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    preds = iso.fit_predict(X)
    # -1 is anomaly, 1 is normal
    df["is_iso_anomaly"] = (preds == -1)
    df["iso_anomaly_score"] = iso.decision_function(X) # lower is more anomalous
    return df

def diagnose_anomaly_reason(row: pd.Series, baseline_stats: Dict[str, Dict[str, float]]) -> str:
    """
    Generate an understandable diagnostic explanation for a flagged anomaly day.
    """
    reasons = []
    
    usage = row.get("total_data_usage_gb", 0)
    downloads = row.get("downloading_gb", 0)
    streaming = row.get("video_streaming_hours", 0)
    devices = row.get("connected_devices_count", 0)
    
    dl_mean = baseline_stats.get("downloading_gb", {}).get("mean", 2.0)
    dl_std = baseline_stats.get("downloading_gb", {}).get("std", 3.0)
    
    st_mean = baseline_stats.get("video_streaming_hours", {}).get("mean", 3.0)
    st_std = baseline_stats.get("video_streaming_hours", {}).get("std", 2.0)
    
    dev_mean = baseline_stats.get("connected_devices_count", {}).get("mean", 8.0)
    
    if downloads > (dl_mean + 2.5 * dl_std) or downloads > 15.0:
        reasons.append(f"Heavy download surge ({downloads:.1f} GB downloaded vs {dl_mean:.1f} GB avg; e.g., large OS update, game install, or media archive)")
        
    if streaming > (st_mean + 2.0 * st_std) or streaming > 6.0:
        reasons.append(f"Excessive video streaming ({streaming:.1f} hrs vs {st_mean:.1f} hrs normal; e.g., binge watching, sports broadcast)")
        
    if devices > (dev_mean + 3):
        reasons.append(f"Unusually high concurrent device count ({int(devices)} devices vs {dev_mean:.0f} average; possible household gathering/guests)")
        
    if not reasons:
        if usage > baseline_stats.get("total_data_usage_gb", {}).get("mean", 15.0) * 1.6:
            reasons.append(f"General multi-activity aggregate peak ({usage:.1f} GB total consumption)")
        else:
            reasons.append(f"Unusually low consumption ({usage:.1f} GB; potential travel, power outage, or absence from home)")
            
    return "; ".join(reasons)

def run_comprehensive_anomaly_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes both IQR and Isolation Forest anomaly screening,
    synthesizes flags, and generates diagnostic explanations.
    """
    df_res = detect_anomalies_iqr(df)
    df_res = detect_anomalies_isolation_forest(df_res)
    
    # Combined flag (either IQR spike or multi-dimensional Isolation Forest flag)
    df_res["is_anomaly"] = df_res["is_iqr_anomaly"] | df_res["is_iso_anomaly"]
    if "day_of_week" not in df_res.columns and "date" in df_res.columns:
        df_res["day_of_week"] = pd.to_datetime(df_res["date"]).dt.strftime("%A")
    
    # Compute baseline statistics for explanations
    base_cols = ["total_data_usage_gb", "downloading_gb", "video_streaming_hours", "connected_devices_count"]
    stats_dict = {}
    for col in base_cols:
        if col in df_res.columns:
            stats_dict[col] = {
                "mean": float(df_res[col].mean()),
                "std": float(df_res[col].std()) if df_res[col].std() > 0 else 1.0,
                "median": float(df_res[col].median())
            }
            
    # Generate human-readable diagnostics for anomalies
    explanations = []
    for _, row in df_res.iterrows():
        if row["is_anomaly"]:
            explanations.append(diagnose_anomaly_reason(row, stats_dict))
        else:
            explanations.append("Normal usage pattern within expected statistical thresholds.")
            
    df_res["anomaly_explanation"] = explanations
    
    anomaly_subset = df_res[df_res["is_anomaly"]].copy()
    
    summary = {
        "total_records_analyzed": len(df_res),
        "total_anomalies_detected": len(anomaly_subset),
        "anomaly_rate_percent": round((len(anomaly_subset) / len(df_res)) * 100, 2),
        "iqr_upper_threshold_gb": round(float(df_res["iqr_upper_threshold"].iloc[0]), 2),
        "max_anomalous_usage_gb": round(float(anomaly_subset["total_data_usage_gb"].max()), 2) if not anomaly_subset.empty else 0.0,
        "academic_note": "Identified anomalies represent statistical consumption variations and activity spikes. They do not imply cybersecurity breaches or unauthorized network intrusions."
    }
    
    return df_res, summary
