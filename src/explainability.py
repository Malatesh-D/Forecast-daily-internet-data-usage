"""
Model Explainability and Interpretability Module.
Computes Tree Feature Importances and SHAP (SHapley Additive exPlanations) values.
Generates human-friendly explanations clarifying feature influence while emphasizing
the distinction between statistical predictive association and causal influence.
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
import shap
from typing import Dict, Any, List, Tuple

from src.preprocessing import prepare_dataset_for_modeling, TARGET_COL
from src.train_models import MODELS_DIR, BEST_MODEL_PATH, METRICS_PATH

CHARTS_DIR = os.path.join("assets", "charts")

def get_tree_feature_importance(model_name: str = "Gradient Boosting") -> pd.DataFrame:
    """
    Extract native feature importances from tree-based models with fallback self-healing.
    """
    model = None
    feature_names = [
        "video_streaming_hours", "downloading_gb", "video_calling_hours",
        "online_gaming_hours", "social_media_hours", "smart_tv_usage_gb",
        "laptop_usage_gb", "mobile_usage_gb", "connected_devices_count",
        "previous_day_usage", "rolling_7_day_average", "is_weekend"
    ]
    
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_meta = json.load(f)
            model_file = metrics_meta["models"][model_name]["file"]
            model = joblib.load(os.path.join(MODELS_DIR, model_file))
        except Exception:
            model = None
            
    if model is None:
        # Fit fresh on the fly if unpickling fails
        from src.database import load_combined_data
        from src.preprocessing import prepare_dataset_for_modeling
        from src.train_models import get_models
        df_base = load_combined_data()
        X_train, _, y_train, _, _, feature_names, _, _ = prepare_dataset_for_modeling(df_base, save_artifacts=False)
        models_dict = get_models()
        model = models_dict.get(model_name, models_dict["Gradient Boosting"])
        model.fit(X_train, y_train)
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
        
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
    # Normalize to percentage
    imp_df["Importance_Pct"] = (imp_df["Importance"] / imp_df["Importance"].sum()) * 100
    imp_df["Importance_Pct"] = imp_df["Importance_Pct"].round(2)
    return imp_df

def compute_shap_summary(sample_size: int = 150) -> Tuple[pd.DataFrame, Any, np.ndarray, List[str]]:
    """
    Compute SHAP values using a representative sample of data.
    Works robustly with TreeExplainer on Gradient Boosting / XGBoost.
    """
    from src.database import load_combined_data
    df = load_combined_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names, _, _ = prepare_dataset_for_modeling(df, save_artifacts=False)
    
    # Use Gradient Boosting or XGBoost for clean tree SHAP explanation
    xgb_path = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
    model = joblib.load(xgb_path)
    
    # Subsample test set for fast and interactive responsiveness
    idx_sample = np.random.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test[idx_sample]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    shap_df["Mean_Abs_SHAP"] = shap_df["Mean_Abs_SHAP"].round(4)
    return shap_df, explainer, shap_values, feature_names

def generate_explainability_charts(output_dir: str = CHARTS_DIR):
    """Generate and save feature importance chart to assets/charts/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    imp_df = get_tree_feature_importance("Gradient Boosting").head(12)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df, x="Importance_Pct", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 12 Features Influencing Daily Internet Usage Prediction (Gradient Boosting)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Feature Importance (%)", fontsize=11)
    ax.set_ylabel("Engineered Feature", fontsize=11)
    
    for i, v in enumerate(imp_df["Importance_Pct"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
        
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("Explainability charts successfully saved.")

if __name__ == "__main__":
    generate_explainability_charts()
