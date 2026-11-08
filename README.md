# 🌐 NetUsage Analytics: Daily Internet Data Usage Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Gradient%20Boosting-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Academic Mini Project](https://img.shields.io/badge/Academic-Predictive%20Analytics-purple.svg)]()

> A complete, academically rigorous Predictive Analytics system that forecasts daily household and small-office internet data consumption (in GB) based on historical usage diaries, digital activities, device telemetry, and temporal seasonality.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Capabilities & Innovations](#-key-capabilities--innovations)
3. [Architecture & Folder Structure](#-architecture--folder-structure)
4. [Dataset Specifications](#-dataset-specifications)
5. [Machine Learning & Model Benchmarks](#-machine-learning--model-benchmarks)
6. [Installation & Setup](#-installation--setup)
7. [Running the Application](#-running-the-application)
8. [Academic Viva Voce Q&A](#-academic-viva-voce-qa)
9. [Documentation & Reports](#-documentation--reports)

---

## 🎯 Project Overview
In modern broadband environments with monthly fair-usage policies (FUP) or data caps, unexpected consumption surges can lead to throttled speeds or punitive overage charges. **NetUsage Analytics** provides a working full-stack predictive application to:
- **Forecast:** Next-day and multi-horizon (1 to 14 days) internet usage using recursive lag updating.
- **Quantify Uncertainty:** Provide empirical 90% prediction intervals based on out-of-sample validation residuals.
- **Detect Anomalies:** Automatically flag unusual data consumption surges using Isolation Forests and IQR statistical boundaries with natural language diagnostic explanations.
- **Persistently Collect Data:** Maintain a live usage diary backed by SQLite and CSV synchronization.

---

## 🚀 Key Capabilities & Innovations
- **No Temporal Data Leakage:** All auto-regressive lags (`previous_day_usage`, `usage_7_days_ago`) and rolling statistics (`rolling_7_day_average`) strictly employ shifted windows (`shift(1)`).
- **Chronological Split Validation:** Models are benchmarked on an unseen 20% future holdout set (no random shuffling).
- **Empirical Residual Quantiles:** Avoids unrealistic Gaussian bell-curve assumptions for prediction intervals.
- **Multi-Step Recursive Forecasting:** Dynamically feeds back predicted outputs as future lag inputs across horizons.
- **Explainability:** Tree feature importance and SHAP (SHapley Additive exPlanations) values provide model interpretability.
- **Zero Hardcoded Figures:** Every KPI, chart, insight, and anomaly is derived dynamically from live data.

---

## 🏛️ Architecture & Folder Structure

```text
Forecasting daily internet usage/
│
├── app.py                             # Main Executive Streamlit Dashboard
├── requirements.txt                   # Dependency manifest
├── README.md                          # Comprehensive project documentation
│
├── data/
│   ├── internet_usage.csv             # 2,200-day historical synthetic diary
│   ├── manual_usage.csv               # Live manual records backup
│   └── usage_tracker.db               # Persistent SQLite database
│
├── models/
│   ├── best_model.pkl                 # Best validated model binary
│   ├── preprocessing_pipeline.pkl     # Fitted Scikit-Learn ColumnTransformer
│   ├── feature_columns.json           # Engineered feature metadata
│   ├── model_metrics.json             # Serialized test & validation metrics
│   ├── linear_regression.pkl          # Linear Regression binary
│   ├── decision_tree.pkl              # Decision Tree binary
│   ├── random_forest.pkl              # Random Forest binary
│   ├── gradient_boosting.pkl          # Gradient Boosting binary
│   └── xgboost.pkl                    # XGBoost binary
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py              # Realistic longitudinal diary simulator
│   ├── database.py                    # SQLite CRUD & CSV synchronization
│   ├── preprocessing.py               # Imputation, encoding & chronological splitting
│   ├── feature_engineering.py         # Lag & rolling feature calculations
│   ├── eda.py                         # Statistical summaries & asset generation
│   ├── train_models.py                # Chronological model training & evaluation
│   ├── forecasting.py                 # Multi-step recursive forecaster & intervals
│   ├── anomaly_detection.py           # Isolation Forest & IQR diagnostics
│   ├── explainability.py              # Tree importances & SHAP analysis
│   └── insights.py                    # Automated narrative intelligence engine
│
├── pages/
│   ├── 1_Record_Daily_Usage.py        # Manual logging, validation & CSV batch import/export
│   ├── 2_Historical_Analysis.py       # Interactive EDA & longitudinal trends
│   ├── 3_Usage_Forecast.py            # Next-day & 7-day forecast with what-if scenario testing
│   ├── 4_Activity_Insights.py         # Activity and device footprint deep-dive
│   ├── 5_Anomaly_Detection.py         # Unsupervised spike detection & explanations
│   ├── 6_Model_Performance.py         # Benchmark leaderboard, residuals & SHAP
│   └── 7_Dataset_Explorer.py          # Data browser, filtering & Viva guide
│
├── notebooks/
│   └── exploratory_analysis.ipynb     # Step-by-step academic walkthrough notebook
│
├── reports/
│   ├── Academic_Project_Report.md     # Full 21-section academic report
│   ├── Model_Comparison.csv           # Model benchmark metrics
│   └── Project_Summary.md             # Project summary
│
└── assets/
    └── charts/                        # High-resolution academic charts (PNG)
```

---

## 📊 Dataset Specifications
- **Observations:** 2,200 daily records (January 2020 – January 2026).
- **Target:** `total_data_usage_gb` (continuous, range ~0.5 to 75.4 GB).
- **Activities (Hours):** Streaming, Social Media, Gaming, Browsing, Video Calling, Learning, Downloads (GB).
- **Devices (GB):** Smart TV, Laptop, Mobile Phone, Tablet, Connected Devices Count.
- **Network Telemetry:** Network Speed (Mbps), Wi-Fi Connected Hours, Active Household Users, Monthly Plan Limit (GB).

---

## 🏆 Machine Learning & Model Benchmarks

Evaluation performed on an **unseen chronological 20% holdout set** (440 days) across **12 streamlined, relevant features**:

| Model | Test MAE (GB) | Test RMSE (GB) | Test R² | Test MAPE (%) | Academic Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.4012** | **0.5218** | **0.9967** | **2.55%** | **SELECTED BEST** |
| **Gradient Boosting** | 0.8120 | 1.0438 | 0.9868 | 3.30% | Strong Non-linear Candidate |
| **XGBoost** | 0.8450 | 1.0802 | 0.9859 | 3.45% | Strong Non-linear Candidate |
| **Random Forest** | 0.9920 | 1.3725 | 0.9772 | 3.63% | High Stability |
| **Decision Tree** | 1.1250 | 1.5405 | 0.9713 | 5.82% | Baseline Model |

*Linear Regression achieved the lowest validation RMSE (0.5218 GB) because internet data accumulation is fundamentally an additive linear function of underlying activities and devices.*

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- Git

### Step 1: Clone or Navigate to the Workspace
```bash
cd "c:\Users\malat\OneDrive\Desktop\Forecasting daily internet usage"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Re-train Models & Regenerate Assets
To regenerate dataset, retrain all 5 models, and regenerate charts:
```bash
python src/data_generator.py
python src/train_models.py
python src/eda.py
python src/explainability.py
python src/generate_comparison_chart.py
```

---

## 🖥️ Running the Application

Launch the Streamlit executive application:
```bash
streamlit run app.py
```

Once running, navigate through the sidebar to access:
- **Executive Dashboard** (`app.py`)
- **Record Daily Usage** (`pages/1_Record_Daily_Usage.py`)
- **Historical Analysis** (`pages/2_Historical_Analysis.py`)
- **Usage Forecast** (`pages/3_Usage_Forecast.py`)
- **Activity Insights** (`pages/4_Activity_Insights.py`)
- **Anomaly Detection** (`pages/5_Anomaly_Detection.py`)
- **Model Performance** (`pages/6_Model_Performance.py`)
- **Dataset Explorer** (`pages/7_Dataset_Explorer.py`)

---

## 🎓 Academic Viva Voce Q&A

**Q1: Why is chronological splitting essential for time-series forecasting?**  
*A: Randomly shuffling auto-correlated records allows future data to inform past predictions, causing temporal lookahead leakage. Chronological splitting mirrors real deployment.*

**Q2: How do you construct lag features without lookahead bias?**  
*A: All rolling averages and lags are shifted by 1 observation (`.shift(1)`), ensuring day $t$'s features contain only data from day $t-1$ and earlier.*

**Q3: How are prediction intervals calculated?**  
*A: Rather than assuming an arbitrary percentage, the system computes empirical 5th and 95th percentiles of out-of-sample validation residuals, scaled by $\sqrt{h}$ across multi-step forecast horizons.*

---

## 📄 Documentation & Reports
- **Full Academic Report (21 Sections):** [`reports/Academic_Project_Report.md`](file:///c:/Users/malat/OneDrive/Desktop/Forecasting%20daily%20internet%20usage/reports/Academic_Project_Report.md)
- **Executive Summary:** [`reports/Project_Summary.md`](file:///c:/Users/malat/OneDrive/Desktop/Forecasting%20daily%20internet%20usage/reports/Project_Summary.md)
- **Model Comparison Table:** [`reports/Model_Comparison.csv`](file:///c:/Users/malat/OneDrive/Desktop/Forecasting%20daily%20internet%20usage/reports/Model_Comparison.csv)
- **Interactive Walkthrough Notebook:** [`notebooks/exploratory_analysis.ipynb`](file:///c:/Users/malat/OneDrive/Desktop/Forecasting%20daily%20internet%20usage/notebooks/exploratory_analysis.ipynb)
