"""
Analytics & Models View - NetUsage Analytics
Consolidated view combining:
1. Historical Analysis & EDA
2. Anomaly Detection & Diagnostics
3. ML Model Benchmarks & Feature Importance
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.database import load_combined_data
from src.anomaly_detection import run_comprehensive_anomaly_detection
from src.explainability import get_tree_feature_importance

def render_analytics_models():
    st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #0f172a; }
        .main-header { font-size: 2.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1rem; color: #64748b; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">📈 Analytics, Anomalies & Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unified module for exploratory data analysis, behavioral anomaly detection, and regression model evaluation.</div>', unsafe_allow_html=True)

    df = load_combined_data()
    if df.empty:
        st.warning("No data found.")
        return

    df["date_dt"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values("date_dt").reset_index(drop=True)
    if "day_of_week" not in df_sorted.columns:
        df_sorted["day_of_week"] = df_sorted["date_dt"].dt.strftime("%A")

    tab_eda, tab_anom, tab_models = st.tabs([
        "📊 Historical Analysis & EDA",
        "🚨 Anomaly Detection",
        "⚙️ Model Performance & Benchmarks"
    ])

    # 1. Historical Analysis Tab
    with tab_eda:
        st.markdown("#### 📊 Exploratory Patterns & Feature Correlations")
        c_e1, c_e2 = st.columns(2)

        with c_e1:
            st.markdown("##### Target Variable Distribution (GB)")
            fig_hist = px.histogram(
                df_sorted,
                x="total_data_usage_gb",
                nbins=35,
                color_discrete_sequence=["#2563eb"],
                labels={"total_data_usage_gb": "Daily Usage (GB)"}
            )
            fig_hist.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c_e2:
            st.markdown("##### Day-of-Week Variation")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            fig_dow = px.box(
                df_sorted,
                x="day_of_week",
                y="total_data_usage_gb",
                category_orders={"day_of_week": day_order},
                color="is_weekend",
                color_discrete_map={0: "#2563eb", 1: "#f59e0b"},
                labels={"day_of_week": "Day", "total_data_usage_gb": "Usage (GB)", "is_weekend": "Weekend"}
            )
            fig_dow.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown("##### 🔥 Feature Correlation Matrix")
        core_cols = [
            "total_data_usage_gb", "video_streaming_hours", "downloading_gb",
            "video_calling_hours", "online_gaming_hours", "social_media_hours",
            "smart_tv_usage_gb", "laptop_usage_gb", "mobile_usage_gb", "connected_devices_count"
        ]
        avail_core = [c for c in core_cols if c in df_sorted.columns]
        corr_matrix = df_sorted[avail_core].corr().round(2)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Correlation with Daily Data Usage"
        )
        fig_corr.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            height=460,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # 2. Anomaly Detection Tab
    with tab_anom:
        st.markdown("#### 🚨 Behavioral Usage Anomaly Detection")
        st.caption("Flags consumption spikes using Isolation Forest and Interquartile Range (IQR).")

        df_ano, summary = run_comprehensive_anomaly_detection(df_sorted)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Screened Days", f"{summary['total_records_analyzed']:,}")
        k2.metric("Flagged Spikes", f"{summary['total_anomalies_detected']}", delta=f"{summary['anomaly_rate_percent']}% rate", delta_color="inverse")
        k3.metric("IQR Upper Boundary", f"{summary['iqr_upper_threshold_gb']:.1f} GB")
        k4.metric("Peak Spike Usage", f"{summary['max_anomalous_usage_gb']:.1f} GB")

        # Scatter
        normal_pts = df_ano[~df_ano["is_anomaly"]]
        anom_pts = df_ano[df_ano["is_anomaly"]]

        fig_ano = go.Figure()
        fig_ano.add_trace(go.Scatter(
            x=normal_pts["date_dt"],
            y=normal_pts["total_data_usage_gb"],
            mode="markers",
            name="Normal Days",
            marker=dict(color="#2563eb", size=4, opacity=0.5)
        ))
        fig_ano.add_trace(go.Scatter(
            x=anom_pts["date_dt"],
            y=anom_pts["total_data_usage_gb"],
            mode="markers",
            name="Anomalous Surge",
            marker=dict(color="#ef4444", size=8, symbol="circle"),
            hovertext=anom_pts["anomaly_explanation"],
            hoverinfo="x+y+text"
        ))
        fig_ano.add_hline(
            y=summary["iqr_upper_threshold_gb"],
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"IQR Upper Threshold ({summary['iqr_upper_threshold_gb']:.1f} GB)"
        )
        fig_ano.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ano, use_container_width=True)

        st.markdown("##### 📋 Anomaly Diagnostic Inspector")
        disp_anom = anom_pts[[
            "date", "day_of_week", "total_data_usage_gb", "downloading_gb",
            "video_streaming_hours", "connected_devices_count", "anomaly_explanation"
        ]].sort_values("total_data_usage_gb", ascending=False)
        disp_anom.columns = ["Date", "Day", "Usage (GB)", "Downloaded (GB)", "Streaming (hrs)", "Devices", "Diagnostic Reason"]
        st.dataframe(disp_anom, use_container_width=True, height=260)

    # 3. Model Performance Tab
    with tab_models:
        st.markdown("#### ⚙️ Regression Model Leaderboard (Chronological Holdout Split)")
        comp_csv = os.path.join("reports", "Model_Comparison.csv")
        if os.path.exists(comp_csv):
            comp_df = pd.read_csv(comp_csv)
            st.dataframe(
                comp_df.style.highlight_min(subset=["Validation MAE (GB)", "Validation RMSE (GB)", "Validation MAPE (%)"], color="#dcfce7")
                                .highlight_max(subset=["Validation R²"], color="#dcfce7"),
                use_container_width=True
            )

            st.write("")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("##### Validation RMSE Comparison (Lower is Better)")
                fig_rmse = px.bar(
                    comp_df,
                    x="Validation RMSE (GB)",
                    y="Model",
                    orientation="h",
                    color="Selected Best",
                    color_discrete_map={"YES": "#10b981", "NO": "#2563eb"},
                    text="Validation RMSE (GB)"
                )
                fig_rmse.update_traces(texttemplate='%{text:.4f} GB', textposition='outside')
                fig_rmse.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    height=280,
                    showlegend=False
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

            with col_m2:
                st.markdown("##### Top Feature Importances (Gradient Boosting)")
                try:
                    imp_df = get_tree_feature_importance("Gradient Boosting").head(8)
                    fig_imp = px.bar(
                        imp_df,
                        x="Importance_Pct",
                        y="Feature",
                        orientation="h",
                        color="Importance_Pct",
                        color_continuous_scale="Blues",
                        labels={"Importance_Pct": "Importance (%)", "Feature": "Feature"}
                    )
                    fig_imp.update_layout(
                        template="plotly_white",
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        height=280,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading feature importances: {str(e)}")
