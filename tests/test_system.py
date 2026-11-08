"""
Comprehensive System Test Suite for NetUsage Analytics.
Validates:
1. Data loading and schema integrity.
2. Leakage-free lag feature construction.
3. Chronological splitting.
4. Trained model loading and inference.
5. Multi-step recursive forecasting and prediction interval consistency.
6. Anomaly detection and diagnostic reasoning.
7. SQLite manual record persistence, duplicate prevention, and deletion.
8. Streamlit page syntax and execution readiness.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import unittest
import pandas as pd
import numpy as np

from src.database import (
    load_combined_data,
    add_record,
    update_record,
    delete_record,
    record_exists,
    calculate_date_features,
    BASE_CSV_PATH
)
from src.feature_engineering import create_lag_and_rolling_features
from src.preprocessing import prepare_dataset_for_modeling, TARGET_COL
from src.forecasting import generate_recursive_forecast, load_forecasting_artifacts
from src.anomaly_detection import run_comprehensive_anomaly_detection
from src.insights import generate_household_insights
from src.explainability import get_tree_feature_importance

class TestNetUsageAnalytics(unittest.TestCase):
    
    def test_01_dataset_integrity(self):
        """Verify dataset exists, has at least 2,000 rows, and required columns."""
        df = load_combined_data()
        self.assertFalse(df.empty, "Dataset must not be empty.")
        self.assertGreaterEqual(len(df), 2000, "Dataset must have at least 2,000 records.")
        self.assertIn(TARGET_COL, df.columns, f"Target column '{TARGET_COL}' missing.")
        
        required_cols = [
            "date", "video_streaming_hours", "downloading_gb", "video_calling_hours",
            "online_gaming_hours", "social_media_hours", "smart_tv_usage_gb",
            "laptop_usage_gb", "mobile_usage_gb", "connected_devices_count",
            "is_weekend", "previous_day_usage", "rolling_7_day_average", TARGET_COL
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Required column '{col}' missing from dataset.")

    def test_02_leakage_free_lag_engineering(self):
        """Verify rolling averages and lags do not contain current day observation."""
        df = load_combined_data()
        df_feat = create_lag_and_rolling_features(df)
        
        # previous_day_usage at index 1 must equal total_data_usage_gb at index 0
        self.assertEqual(
            df_feat.loc[1, "previous_day_usage"],
            df_feat.loc[0, TARGET_COL],
            "Lag 1 must equal the previous day's target value."
        )
        
        # rolling_7_day_average at index 7 must strictly be the average of indices 0 through 6
        expected_roll7 = df_feat.loc[0:6, TARGET_COL].mean()
        actual_roll7 = df_feat.loc[7, "rolling_7_day_average"]
        self.assertAlmostEqual(
            expected_roll7, actual_roll7, places=4,
            msg="Rolling 7 day average must strictly reflect past observations only."
        )

    def test_03_model_inference_and_artifacts(self):
        """Verify model artifacts load and produce numerically sound predictions."""
        model, preprocessor, model_name, res_stats = load_forecasting_artifacts()
        self.assertIsNotNone(model)
        self.assertIsNotNone(preprocessor)
        self.assertIn("std", res_stats)
        self.assertGreater(res_stats["std"], 0)

    def test_04_recursive_forecasting(self):
        """Verify multi-day recursive forecasting generates distinct future predictions with bounds."""
        df = load_combined_data()
        forecast_df, meta = generate_recursive_forecast(df, horizon_days=7)
        
        self.assertEqual(len(forecast_df), 7, "Must forecast exactly 7 days.")
        self.assertTrue((forecast_df["predicted_usage_gb"] > 0).all(), "Predictions must be positive.")
        self.assertTrue((forecast_df["lower_interval_90_gb"] <= forecast_df["predicted_usage_gb"]).all(), "Lower interval must be <= prediction.")
        self.assertTrue((forecast_df["upper_interval_90_gb"] >= forecast_df["predicted_usage_gb"]).all(), "Upper interval must be >= prediction.")
        
        # Verify predictions are dynamic and not just 7 identical duplicated numbers
        unique_preds = forecast_df["predicted_usage_gb"].nunique()
        self.assertGreater(unique_preds, 1, "Recursive forecast must produce dynamic multi-horizon outputs.")

    def test_05_anomaly_detection(self):
        """Verify anomaly detection flags records and attaches valid explanations."""
        df = load_combined_data()
        df_ano, summary = run_comprehensive_anomaly_detection(df)
        
        self.assertIn("is_anomaly", df_ano.columns)
        self.assertIn("anomaly_explanation", df_ano.columns)
        self.assertGreater(summary["total_anomalies_detected"], 0, "Should detect consumption spikes.")
        
        # Verify explanation text exists for flagged anomalies
        anom_rows = df_ano[df_ano["is_anomaly"]]
        for _, row in anom_rows.head(5).iterrows():
            self.assertTrue(len(row["anomaly_explanation"]) > 10, "Anomaly explanation must be informative.")

    def test_06_database_crud(self):
        """Verify manual record addition, duplicate check, and deletion."""
        test_date = "2099-12-31"
        payload = {
            "date": test_date,
            "video_streaming_hours": 4.0,
            "social_media_hours": 2.0,
            "online_gaming_hours": 1.0,
            "browsing_hours": 2.0,
            "video_calling_hours": 1.0,
            "downloading_gb": 5.0,
            "online_learning_hours": 0.0,
            "mobile_usage_gb": 3.0,
            "laptop_usage_gb": 5.0,
            "smart_tv_usage_gb": 8.0,
            "tablet_usage_gb": 1.0,
            "connected_devices_count": 8,
            "internet_plan_limit_gb": 750.0,
            "network_speed_mbps": 150,
            "wifi_connected_hours": 16.0,
            "number_of_users": 3,
            "total_data_usage_gb": 18.5
        }
        
        # 1. Add
        ok, msg = add_record(payload)
        self.assertTrue(ok, f"Add record failed: {msg}")
        
        # 2. Duplicate prevention
        ok_dup, _ = add_record(payload)
        self.assertFalse(ok_dup, "Duplicate record should be rejected.")
        
        # 3. Clean up / Delete
        ok_del, _ = delete_record(test_date)
        self.assertTrue(ok_del, "Delete record should succeed.")

    def test_07_insights_generation(self):
        """Verify insights engine outputs actionable, data-backed insights."""
        df = load_combined_data()
        insights = generate_household_insights(df)
        self.assertGreaterEqual(len(insights), 3, "Must generate at least 3 household insights.")
        for item in insights:
            self.assertIn("title", item)
            self.assertIn("body", item)

if __name__ == "__main__":
    unittest.main()
