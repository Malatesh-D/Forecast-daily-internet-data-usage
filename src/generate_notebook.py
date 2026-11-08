"""
Generate Academic Jupyter Notebook for Exploratory Data Analysis & Modeling Walkthrough.
"""
import os
import json

def create_notebook():
    notebook_path = os.path.join("notebooks", "exploratory_analysis.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Daily Internet Data Usage Forecasting Using Machine Learning\n",
                "## Academic Mini-Project: Exploratory Data Analysis & Predictive Modeling\n",
                "\n",
                "**Author:** Data Science Research Team  \n",
                "**System Name:** NetUsage Analytics  \n",
                "**Target Variable:** `total_data_usage_gb` (Regression)\n",
                "\n",
                "---\n",
                "### Table of Contents\n",
                "1. [Environment Setup & Data Ingestion](#1.-Environment-Setup--Data-Ingestion)\n",
                "2. [Exploratory Data Analysis (EDA)](#2.-Exploratory-Data-Analysis-(EDA))\n",
                "3. [Feature Engineering & Leakage-Free Lag Generation](#3.-Feature-Engineering--Leakage-Free-Lag-Generation)\n",
                "4. [Chronological Train-Test Split & Pipeline Preprocessing](#4.-Chronological-Train-Test-Split--Pipeline-Preprocessing)\n",
                "5. [Machine Learning Model Training & Benchmarking](#5.-Machine-Learning-Model-Training--Benchmarking)\n",
                "6. [Multi-Day Recursive Forecasting Simulation](#6.-Multi-Day-Recursive-Forecasting-Simulation)\n",
                "7. [Unsupervised Anomaly Detection](#7.-Unsupervised-Anomaly-Detection)\n",
                "8. [Academic Conclusions & Viva Summary](#8.-Academic-Conclusions--Viva-Summary)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Ingestion and Library Imports\n",
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add root directory to sys.path\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "\n",
                "from src.database import load_combined_data\n",
                "from src.preprocessing import prepare_dataset_for_modeling\n",
                "from src.train_models import get_models, evaluate_predictions\n",
                "from src.anomaly_detection import run_comprehensive_anomaly_detection\n",
                "\n",
                "# Visual configurations\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 5)\n",
                "print('Setup complete. All modules imported.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Ingesting the Longitudinal Internet Usage Diary Dataset\n",
                "We load our historical consumption dataset spanning 2,200 daily records (2020 to 2026)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = load_combined_data(base_csv_path='../data/internet_usage.csv', db_path='../data/usage_tracker.db')\n",
                "print(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')\n",
                "display(df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Exploratory Data Analysis (EDA)\n",
                "Examining the statistical properties and distribution of `total_data_usage_gb`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(df.describe().round(2))\n",
                "\n",
                "plt.figure(figsize=(9, 4))\n",
                "sns.histplot(df['total_data_usage_gb'], kde=True, color='#2563eb', bins=35)\n",
                "plt.title('Distribution of Daily Internet Data Usage (GB)')\n",
                "plt.xlabel('Total Usage (GB)')\n",
                "plt.ylabel('Frequency')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Chronological Train-Test Split and Preprocessing Pipeline\n",
                "We apply an 80/20 chronological split to avoid time-series lookahead leakage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train, X_test, y_train, y_test, preprocessor, feature_names, train_df, test_df = prepare_dataset_for_modeling(df, save_artifacts=False)\n",
                "print(f'Train cohort: {X_train.shape[0]} days ({train_df[\"date\"].min()} to {train_df[\"date\"].max()})')\n",
                "print(f'Test cohort:  {X_test.shape[0]} days ({test_df[\"date\"].min()} to {test_df[\"date\"].max()})')\n",
                "print(f'Total engineered features: {len(feature_names)}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Training and Evaluating Candidate Regression Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = get_models()\n",
                "results = []\n",
                "\n",
                "for name, model in models.items():\n",
                "    model.fit(X_train, y_train)\n",
                "    preds = model.predict(X_test)\n",
                "    metrics = evaluate_predictions(y_test, preds)\n",
                "    results.append({\n",
                "        'Model': name,\n",
                "        'Test MAE (GB)': metrics['MAE'],\n",
                "        'Test RMSE (GB)': metrics['RMSE'],\n",
                "        'Test R2': metrics['R2'],\n",
                "        'Test MAPE (%)': metrics['MAPE']\n",
                "    })\n",
                "\n",
                "results_df = pd.DataFrame(results).sort_values('Test RMSE (GB)')\n",
                "display(results_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Anomaly Detection Demonstration\n",
                "Testing Isolation Forest and IQR anomaly detection on consumption spikes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_anomalies, summary = run_comprehensive_anomaly_detection(df)\n",
                "print('Anomaly Detection Summary:', summary)\n",
                "display(df_anomalies[df_anomalies['is_anomaly']][['date', 'total_data_usage_gb', 'downloading_gb', 'anomaly_explanation']].head(10))"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.13.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Jupyter notebook created at {notebook_path}")

if __name__ == "__main__":
    create_notebook()
