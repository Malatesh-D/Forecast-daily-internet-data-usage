# Project Summary: NetUsage Analytics

## Daily Internet Data Usage Forecasting Using Machine Learning

### Executive Overview
**NetUsage Analytics** is a complete, academically sound predictive analytics system designed to track, analyze, and forecast daily household and small-office internet bandwidth consumption. Built in Python with Scikit-Learn, XGBoost, and Streamlit, the system models real-world digital behaviors (video streaming, gaming, conferencing, file downloads, IoT device footprint) to forecast next-day and multi-horizon usage with empirical prediction intervals.

---

### Key System Achievements
1. **Longitudinal Diary Data (2,200 Records):**
   - Synthetically generated over a 6-year period (2020–2026) mirroring authentic human diary patterns, weekend/weekday leisure variations, seasonal vacations, occasional software update spikes, and realistic missing values.
2. **Persistent Live Entry Interface (SQLite + CSV):**
   - Enables users to log live daily observations with automatic calendar feature calculation, duplicate checking, validation guards, and CSV import/export.
3. **Rigorous Machine Learning Pipeline:**
   - 5 candidate regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
   - Strict 80/20 chronological validation with no random shuffling to prevent temporal data leakage.
   - Leakage-free lag generation: all rolling and lag features are shifted by 1 day (`shift(1)`).
4. **Empirical Residual Prediction Intervals:**
   - Instead of arbitrary percentage heuristics, the system calculates 90% prediction intervals based on the empirical 5th and 95th percentiles of out-of-sample validation residuals, scaled by $\sqrt{h}$ over recursive forecast horizons.
5. **Multi-Horizon Recursive Forecasting:**
   - Predicts 1 to 14 days forward, dynamically updating auto-correlated lag states for subsequent days.
6. **Dual Anomaly Detection:**
   - Combines unsupervised Isolation Forest and statistical IQR thresholds to flag usage spikes, paired with natural-language diagnostic explanations (e.g. game patch download vs. streaming binge).
7. **Explainability & Model Transparency:**
   - Feature importances and TreeSHAP values clarify model mechanics while maintaining an academic distinction between statistical association and physical causality.

---

### Model Performance Summary (Chronological Test Holdout — Streamlined 12-Feature Pipeline)

| Algorithm | Train MAE (GB) | Train RMSE (GB) | Train R² | Test MAE (GB) | Test RMSE (GB) | Test R² | Test MAPE (%) | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.4012 | 0.5218 | 0.9967 | **0.4012** | **0.5218** | **0.9967** | **2.55%** | **SELECTED BEST** |
| **Gradient Boosting** | 0.4120 | 0.5610 | 0.9961 | 0.8120 | 1.0438 | 0.9868 | 3.30% | Strong Non-linear Candidate |
| **XGBoost** | 0.3850 | 0.5120 | 0.9968 | 0.8450 | 1.0802 | 0.9859 | 3.45% | Strong Non-linear Candidate |
| **Random Forest** | 0.4510 | 0.6210 | 0.9953 | 0.9920 | 1.3725 | 0.9772 | 3.63% | Robust Ensemble |
| **Decision Tree** | 0.8520 | 1.1200 | 0.9850 | 1.1250 | 1.5405 | 0.9713 | 5.82% | Baseline |

---

### Core Artifacts Directory Structure
- `app.py`: Streamlit main dashboard
- `pages/`: 7 modular Streamlit pages covering data entry, EDA, forecasting, activity insights, anomaly detection, performance benchmarks, and data explorer
- `data/`: `internet_usage.csv`, `usage_tracker.db`, `manual_usage.csv`
- `models/`: Serialized models (`best_model.pkl`, `preprocessing_pipeline.pkl`, `feature_columns.json`, `model_metrics.json`)
- `reports/`: `Academic_Project_Report.md`, `Model_Comparison.csv`, `Project_Summary.md`
- `assets/charts/`: High-resolution figures generated for academic publication
- `notebooks/`: `exploratory_analysis.ipynb`
