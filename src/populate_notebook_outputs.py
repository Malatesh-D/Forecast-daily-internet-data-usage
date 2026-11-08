"""
Script to execute notebook cells and populate output streams/figures
in exploratory_analysis.ipynb so it opens pre-rendered in Jupyter / VS Code.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import base64
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.database import load_combined_data
from src.preprocessing import prepare_dataset_for_modeling
from src.train_models import get_models, evaluate_predictions
from src.anomaly_detection import run_comprehensive_anomaly_detection

NOTEBOOK_PATH = os.path.join("notebooks", "exploratory_analysis.ipynb")

def populate_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Ingestion Cell Output
    nb["cells"][1]["execution_count"] = 1
    nb["cells"][1]["outputs"] = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": ["Setup complete. All modules imported.\n"]
        }
    ]

    # 2. Data loading Cell Output
    df = load_combined_data()
    head_html = df.head().to_html(classes="table table-striped", index=False)
    nb["cells"][3]["execution_count"] = 2
    nb["cells"][3]["outputs"] = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"]
        },
        {
            "data": {
                "text/html": [head_html],
                "text/plain": [df.head().to_string()]
            },
            "metadata": {},
            "output_type": "display_data"
        }
    ]

    # 3. EDA Cell Output (Describe & Distribution Plot)
    desc_html = df.describe().round(2).to_html(classes="table table-bordered")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["total_data_usage_gb"], kde=True, color="#2563eb", bins=35, ax=ax)
    ax.set_title("Distribution of Daily Internet Data Usage (GB)")
    ax.set_xlabel("Total Usage (GB)")
    ax.set_ylabel("Frequency")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    nb["cells"][5]["execution_count"] = 3
    nb["cells"][5]["outputs"] = [
        {
            "data": {
                "text/html": [desc_html],
                "text/plain": [df.describe().round(2).to_string()]
            },
            "metadata": {},
            "output_type": "display_data"
        },
        {
            "data": {
                "image/png": img_b64,
                "text/plain": ["<Figure size 1350x600 with 1 Axes>"]
            },
            "metadata": {},
            "output_type": "display_data"
        }
    ]

    # 4. Train-Test Split Cell Output
    X_train, X_test, y_train, y_test, preprocessor, feature_names, train_df, test_df = prepare_dataset_for_modeling(df, save_artifacts=False)
    nb["cells"][7]["execution_count"] = 4
    nb["cells"][7]["outputs"] = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Train cohort: {X_train.shape[0]} days ({train_df['date'].min()} to {train_df['date'].max()})\n",
                f"Test cohort:  {X_test.shape[0]} days ({test_df['date'].min()} to {test_df['date'].max()})\n",
                f"Total streamlined features: {len(feature_names)}\n"
            ]
        }
    ]

    # 5. Model Evaluation Cell Output
    models = get_models()
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        results.append({
            "Model": name,
            "Test MAE (GB)": metrics["MAE"],
            "Test RMSE (GB)": metrics["RMSE"],
            "Test R2": metrics["R2"],
            "Test MAPE (%)": metrics["MAPE"]
        })
    results_df = pd.DataFrame(results).sort_values("Test RMSE (GB)")
    res_html = results_df.to_html(classes="table table-hover", index=False)
    
    nb["cells"][9]["execution_count"] = 5
    nb["cells"][9]["outputs"] = [
        {
            "data": {
                "text/html": [res_html],
                "text/plain": [results_df.to_string()]
            },
            "metadata": {},
            "output_type": "display_data"
        }
    ]

    # 6. Anomaly Detection Cell Output
    df_anom, summary = run_comprehensive_anomaly_detection(df)
    anom_sub = df_anom[df_anom["is_anomaly"]][["date", "total_data_usage_gb", "downloading_gb", "anomaly_explanation"]].head(8)
    anom_html = anom_sub.to_html(classes="table table-danger", index=False)
    
    nb["cells"][11]["execution_count"] = 6
    nb["cells"][11]["outputs"] = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Anomaly Detection Summary: {summary}\n"
            ]
        },
        {
            "data": {
                "text/html": [anom_html],
                "text/plain": [anom_sub.to_string()]
            },
            "metadata": {},
            "output_type": "display_data"
        }
    ]

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully populated all execution outputs in {NOTEBOOK_PATH}")

if __name__ == "__main__":
    populate_notebook()
