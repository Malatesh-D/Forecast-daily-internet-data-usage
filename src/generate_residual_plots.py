"""
Generate Model Performance and Residual Plots for Section 4 of the Academic Report.
Plots:
1. Actual vs Predicted Values (Scatter with 45-degree reference line)
2. Residual Distribution and Residuals vs Predicted Plot
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from src.database import load_combined_data
from src.preprocessing import prepare_dataset_for_modeling, TARGET_COL
from src.train_models import BEST_MODEL_PATH

CHARTS_DIR = os.path.join("assets", "charts")

def generate_residual_and_performance_plots():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    df = load_combined_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names, train_df, test_df = prepare_dataset_for_modeling(df, save_artifacts=False)
    
    model = joblib.load(BEST_MODEL_PATH)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Actual vs Predicted
    ax1.scatter(y_test, y_pred, alpha=0.6, color="#2563eb", edgecolors="w", s=40, label="Holdout Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="#dc2626", linestyle="--", linewidth=2, label="Ideal Fit (y = ŷ)")
    ax1.set_title("Actual vs. Predicted Total Daily Usage (Holdout Set)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Actual Usage (GB)", fontsize=11)
    ax1.set_ylabel("Predicted Usage (GB)", fontsize=11)
    ax1.legend(loc="upper left")
    
    # 2. Residuals vs Predicted
    ax2.scatter(y_pred, residuals, alpha=0.6, color="#059669", edgecolors="w", s=40)
    ax2.axhline(0, color="#dc2626", linestyle="--", linewidth=1.8)
    q05 = np.percentile(residuals, 5)
    q95 = np.percentile(residuals, 95)
    ax2.axhline(q05, color="#f59e0b", linestyle=":", label=f"5th Percentile ({q05:.2f} GB)")
    ax2.axhline(q95, color="#f59e0b", linestyle=":", label=f"95th Percentile ({q95:.2f} GB)")
    ax2.set_title("Residuals vs. Fitted Values (Homoscedasticity Check)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Predicted Usage (GB)", fontsize=11)
    ax2.set_ylabel("Residual Error (GB: Actual - Predicted)", fontsize=11)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    output_path = os.path.join(CHARTS_DIR, "residual_plots.png")
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"Residual plots saved at {output_path}")

if __name__ == "__main__":
    generate_residual_and_performance_plots()
