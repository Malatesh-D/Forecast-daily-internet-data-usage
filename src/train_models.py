"""
Model Training and Evaluation Module.
Trains Linear Regression, Decision Tree, Random Forest, Gradient Boosting,
and XGBoost Regressors using chronological validation.
Computes MAE, MSE, RMSE, R2, and MAPE, saves model comparison artifacts,
and records empirical validation residual quantiles for prediction intervals.
"""

import os
import sys

# Ensure root path is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from src.database import load_combined_data
from src.preprocessing import prepare_dataset_for_modeling, TARGET_COL

MODELS_DIR = "models"
REPORTS_DIR = "reports"
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
COMPARISON_CSV_PATH = os.path.join(REPORTS_DIR, "Model_Comparison.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

def get_models() -> Dict[str, Any]:
    """Define dictionary of candidate regression models with solid hyperparameters."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
            learning_rate=0.07,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="rmse"
        )
    }

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate key regression evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100) # as percentage
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2)
    }

def train_and_evaluate_all():
    """
    Main training pipeline:
    - Loads dataset
    - Chronologically prepares train and test splits
    - Trains all candidate models
    - Computes training and test metrics
    - Computes empirical residual distributions for forecast prediction intervals
    - Identifies and saves the best model based on validation RMSE
    - Saves all serialized model files and metrics JSON/CSV
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("Loading data and preparing features...")
    df = load_combined_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names, train_df, test_df = prepare_dataset_for_modeling(df)
    
    models = get_models()
    results = {}
    fitted_models = {}
    test_predictions = {}
    
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_metrics = evaluate_predictions(y_train, train_pred)
        test_metrics = evaluate_predictions(y_test, test_pred)
        
        # Calculate residuals for uncertainty estimation
        residuals = y_test - test_pred
        q05 = float(np.percentile(residuals, 5))
        q25 = float(np.percentile(residuals, 25))
        q50 = float(np.percentile(residuals, 50))
        q75 = float(np.percentile(residuals, 75))
        q95 = float(np.percentile(residuals, 95))
        res_std = float(np.std(residuals))
        
        # Save individual model
        clean_filename = name.lower().replace(" ", "_") + ".pkl"
        model_save_path = os.path.join(MODELS_DIR, clean_filename)
        joblib.dump(model, model_save_path)
        
        fitted_models[name] = model
        test_predictions[name] = test_pred
        
        results[name] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "residual_stats": {
                "std": round(res_std, 4),
                "q05": round(q05, 4),
                "q25": round(q25, 4),
                "q50": round(q50, 4),
                "q75": round(q75, 4),
                "q95": round(q95, 4)
            },
            "file": clean_filename
        }
        
        print(f"  {name} -> Test RMSE: {test_metrics['RMSE']:.4f} GB | Test R²: {test_metrics['R2']:.4f} | Test MAPE: {test_metrics['MAPE']:.2f}%")
        
    # Select best model based on lowest test RMSE
    best_name = min(results.keys(), key=lambda k: results[k]["test_metrics"]["RMSE"])
    best_model = fitted_models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    
    # Save best model name in metrics
    summary_payload = {
        "best_model_name": best_name,
        "models": results,
        "evaluation_criteria": "Lowest Test RMSE on chronological holdout set"
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(summary_payload, f, indent=4)
        
    # Export Model Comparison CSV for Academic Report
    comparison_rows = []
    for m_name, d in results.items():
        comparison_rows.append({
            "Model": m_name,
            "Train MAE (GB)": d["train_metrics"]["MAE"],
            "Train RMSE (GB)": d["train_metrics"]["RMSE"],
            "Train R²": d["train_metrics"]["R2"],
            "Validation MAE (GB)": d["test_metrics"]["MAE"],
            "Validation MSE (GB²)": d["test_metrics"]["MSE"],
            "Validation RMSE (GB)": d["test_metrics"]["RMSE"],
            "Validation R²": d["test_metrics"]["R2"],
            "Validation MAPE (%)": d["test_metrics"]["MAPE"],
            "Selected Best": "YES" if m_name == best_name else "NO"
        })
    
    comp_df = pd.DataFrame(comparison_rows).sort_values("Validation RMSE (GB)")
    comp_df.to_csv(COMPARISON_CSV_PATH, index=False)
    
    print(f"\n==========================================")
    print(f"Best Model Selected: {best_name}")
    print(f"Validation RMSE: {results[best_name]['test_metrics']['RMSE']} GB")
    print(f"Validation R²:   {results[best_name]['test_metrics']['R2']}")
    print(f"Artifacts successfully saved to {MODELS_DIR}/ and {REPORTS_DIR}/")
    print(f"==========================================")
    
    return summary_payload, comp_df

if __name__ == "__main__":
    train_and_evaluate_all()
