"""
Exploratory Data Analysis (EDA) Module.
Provides statistical calculations, distribution summaries, correlation matrices,
and high-resolution chart generators saved to assets/charts/ for academic reporting.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for server generation
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

CHARTS_DIR = os.path.join("assets", "charts")

def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute structural metrics, memory usage, duplicates, and missing counts."""
    total_records = len(df)
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    duplicate_dates = int(df.duplicated(subset=["date"]).sum())
    
    date_min = str(pd.to_datetime(df["date"]).min().date()) if "date" in df.columns else "N/A"
    date_max = str(pd.to_datetime(df["date"]).max().date()) if "date" in df.columns else "N/A"
    
    return {
        "total_records": total_records,
        "total_columns": df.shape[1],
        "date_range": (date_min, date_max),
        "total_missing_values": total_missing,
        "missing_per_column": {k: v for k, v in missing_counts.items() if v > 0},
        "duplicate_records": duplicate_dates,
        "manual_entries": int(df["is_manual"].sum()) if "is_manual" in df.columns else 0
    }

def get_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive summary table of numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[num_cols].describe().T
    stats["median"] = df[num_cols].median()
    stats["skewness"] = df[num_cols].skew()
    stats["kurtosis"] = df[num_cols].kurtosis()
    return stats.round(3)

def generate_eda_charts(df: pd.DataFrame, output_dir: str = CHARTS_DIR):
    """
    Generate and save publication-grade academic visualization figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})
    
    df_plot = df.copy()
    if "date" in df_plot.columns:
        df_plot["date"] = pd.to_datetime(df_plot["date"])
        df_plot = df_plot.sort_values("date")
        
    # 1. Target Distribution & KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_plot["total_data_usage_gb"], kde=True, color="#2563eb", ax=ax, bins=35)
    mean_val = df_plot["total_data_usage_gb"].mean()
    median_val = df_plot["total_data_usage_gb"].median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.8, label=f"Mean ({mean_val:.2f} GB)")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=2.0, label=f"Median ({median_val:.2f} GB)")
    ax.set_title("Distribution of Daily Internet Data Usage (GB)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Data Usage (GB)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(frameon=True)
    fig.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=250)
    plt.close(fig)
    
    # 2. Correlation Heatmap
    core_features = [
        "total_data_usage_gb", "video_streaming_hours", "downloading_gb",
        "video_calling_hours", "online_gaming_hours", "social_media_hours",
        "browsing_hours", "smart_tv_usage_gb", "laptop_usage_gb",
        "mobile_usage_gb", "connected_devices_count", "number_of_users"
    ]
    avail_core = [c for c in core_features if c in df_plot.columns]
    corr_matrix = df_plot[avail_core].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix with Internet Data Usage", fontsize=13, fontweight="bold", pad=12)
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=250)
    plt.close(fig)
    
    # 3. Weekly Consumption Patterns (Day of Week Boxplot)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x="day_of_week", y="total_data_usage_gb", data=df_plot, order=day_order, hue="day_of_week", palette="Blues", legend=False, ax=ax)
    ax.set_title("Daily Internet Usage Variation Across Days of the Week", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Day of the Week", fontsize=11)
    ax.set_ylabel("Total Data Usage (GB)", fontsize=11)
    fig.savefig(os.path.join(output_dir, "weekly_trend.png"), dpi=250)
    plt.close(fig)
    
    # 4. Device Share Breakdown (Mean Usage)
    device_cols = ["smart_tv_usage_gb", "laptop_usage_gb", "mobile_usage_gb", "tablet_usage_gb"]
    avail_dev = [c for c in device_cols if c in df_plot.columns]
    if avail_dev:
        mean_dev = df_plot[avail_dev].mean()
        labels = [c.replace("_usage_gb", "").replace("_", " ").title() for c in avail_dev]
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.pie(mean_dev, labels=labels, autopct="%1.1f%%", startangle=140, colors=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"],
               wedgeprops=dict(width=0.4, edgecolor='w'))
        ax.set_title("Average Proportion of Internet Usage by Device", fontsize=13, fontweight="bold", pad=12)
        fig.savefig(os.path.join(output_dir, "device_distribution.png"), dpi=250)
        plt.close(fig)
        
    # 5. Long-term Historical Daily Trend
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_plot["date"], df_plot["total_data_usage_gb"], color="#93c5fd", alpha=0.5, label="Daily Usage (Observed)")
    rolling_30 = df_plot["total_data_usage_gb"].rolling(30, min_periods=7).mean()
    ax.plot(df_plot["date"], rolling_30, color="#1d4ed8", linewidth=2.2, label="30-Day Moving Average")
    ax.set_title("Longitudinal Internet Consumption Diary Trend (2020 - 2026)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Data Usage (GB)", fontsize=11)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(output_dir, "daily_trend.png"), dpi=250)
    plt.close(fig)
    
    print(f"Academic EDA charts successfully generated in '{output_dir}'.")

if __name__ == "__main__":
    from src.database import load_combined_data
    df = load_combined_data()
    print("Generating EDA figures...")
    generate_eda_charts(df)
