"""
Generate Model Comparison Chart for Academic Documentation and Dashboard.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_model_comparison_chart():
    csv_path = os.path.join("reports", "Model_Comparison.csv")
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    output_path = os.path.join("assets", "charts", "model_comparison.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Validation RMSE
    colors = ["#10b981" if x == "YES" else "#3b82f6" for x in df["Selected Best"]]
    bars1 = ax1.barh(df["Model"], df["Validation RMSE (GB)"], color=colors, height=0.55)
    ax1.set_title("Validation RMSE Comparison (Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Root Mean Squared Error (GB)", fontsize=11)
    ax1.invert_yaxis()
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.4f} GB", va="center", fontsize=10, fontweight="bold")
        
    # 2. Validation R2 Score
    bars2 = ax2.barh(df["Model"], df["Validation R²"], color=colors, height=0.55)
    ax2.set_title("Validation R² Score Comparison (Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Coefficient of Determination (R²)", fontsize=11)
    ax2.set_xlim(0.98, 1.0)
    ax2.invert_yaxis()
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.0005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"Model comparison chart saved at {output_path}")

if __name__ == "__main__":
    generate_model_comparison_chart()
