"""
Forecast View - NetUsage Analytics
Crisp white theme multi-horizon recursive forecasting with prediction intervals,
what-if scenario testing, and quota exhaustion monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.database import load_combined_data
from src.forecasting import generate_recursive_forecast

def render_forecast():
    st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #0f172a; }
        .main-header { font-size: 2.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1rem; color: #64748b; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🔮 Internet Usage Forecasting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-horizon recursive predictions with empirical 90% prediction intervals and quota compliance analysis.</div>', unsafe_allow_html=True)

    df = load_combined_data()
    if df.empty:
        st.warning("No data found.")
        return

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        horizon = st.slider("Forecast Horizon (Days)", min_value=1, max_value=14, value=7, step=1)
    with col_ctrl2:
        enable_sim = st.checkbox("🧪 Enable Tomorrow What-If Scenario Simulator", value=False)

    custom_inputs = None
    if enable_sim:
        st.caption("Customize your anticipated hours and downloads for tomorrow:")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            sim_stream = st.slider("Streaming (hrs)", 0.0, 15.0, 4.5, 0.5)
        with sc2:
            sim_dl = st.slider("Downloads (GB)", 0.0, 60.0, 8.0, 1.0)
        with sc3:
            sim_call = st.slider("Video Calls (hrs)", 0.0, 8.0, 1.5, 0.5)
        with sc4:
            sim_gaming = st.slider("Gaming (hrs)", 0.0, 10.0, 2.0, 0.5)

        custom_inputs = {
            "video_streaming_hours": sim_stream,
            "downloading_gb": sim_dl,
            "video_calling_hours": sim_call,
            "online_gaming_hours": sim_gaming,
            "social_media_hours": 2.0,
            "smart_tv_usage_gb": 6.5,
            "laptop_usage_gb": 4.5,
            "mobile_usage_gb": 2.5,
            "connected_devices_count": 8
        }

    # Generate Forecast
    forecast_df, f_meta = generate_recursive_forecast(
        df,
        horizon_days=horizon,
        custom_next_day_inputs=custom_inputs
    )

    next_day = forecast_df.iloc[0]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"Tomorrow ({next_day['date']})",
        f"{next_day['predicted_usage_gb']:.1f} GB",
        f"90% Range: {next_day['lower_interval_90_gb']:.1f} - {next_day['upper_interval_90_gb']:.1f} GB",
        delta_color="off"
    )
    m2.metric(
        f"{horizon}-Day Total",
        f"{forecast_df['predicted_usage_gb'].sum():.1f} GB",
        f"Avg {forecast_df['predicted_usage_gb'].mean():.1f} GB/day"
    )
    m3.metric(
        "Monthly Plan Limit",
        f"{f_meta['monthly_plan_limit_gb']:.0f} GB",
        f"Fair Quota: {f_meta['fair_daily_quota_gb']:.1f} GB/day"
    )
    m4.metric(
        "Forecasting Engine",
        f_meta["model_used"]
    )

    st.write("")

    # Trajectory Plot
    st.markdown("#### 📈 Longitudinal Forecast Trajectory")
    recent_hist = df.tail(30).copy()
    recent_hist["date_dt"] = pd.to_datetime(recent_hist["date"])
    f_dates = pd.to_datetime(forecast_df["date"])

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=recent_hist["date_dt"],
        y=recent_hist["total_data_usage_gb"],
        mode="lines+markers",
        name="Actual (Past 30 Days)",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=4)
    ))

    # Forecast
    bridge_x = [recent_hist["date_dt"].iloc[-1]] + list(f_dates)
    bridge_y = [recent_hist["total_data_usage_gb"].iloc[-1]] + list(forecast_df["predicted_usage_gb"])

    fig.add_trace(go.Scatter(
        x=bridge_x,
        y=bridge_y,
        mode="lines+markers",
        name=f"Recursive Forecast ({f_meta['model_used']})",
        line=dict(color="#10b981", width=2.5, dash="solid"),
        marker=dict(size=6, symbol="diamond", color="#059669")
    ))

    # Interval Bounds
    fig.add_trace(go.Scatter(
        x=list(f_dates) + list(f_dates)[::-1],
        y=list(forecast_df["upper_interval_90_gb"]) + list(forecast_df["lower_interval_90_gb"])[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="90% Empirical Prediction Interval"
    ))

    # Fair Quota Line
    fig.add_hline(
        y=f_meta["fair_daily_quota_gb"],
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"Fair Daily Quota: {f_meta['fair_daily_quota_gb']:.1f} GB"
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Usage (GB)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast Table
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("##### 📋 Day-by-Day Forecast Schedule")
        disp = forecast_df.copy()
        disp["Predicted (GB)"] = disp["predicted_usage_gb"].round(2)
        disp["90% Lower (GB)"] = disp["lower_interval_90_gb"].round(2)
        disp["90% Upper (GB)"] = disp["upper_interval_90_gb"].round(2)
        disp["Quota Status"] = np.where(
            disp["Predicted (GB)"] > f_meta["fair_daily_quota_gb"],
            "⚠️ Over Quota",
            "✅ Within Quota"
        )
        st.dataframe(disp[["date", "day_of_week", "Predicted (GB)", "90% Lower (GB)", "90% Upper (GB)", "Quota Status"]], use_container_width=True)

    with col_t2:
        st.markdown("##### 🛡️ Quota Risk Analysis")
        proj_tot = forecast_df["predicted_usage_gb"].sum()
        fair_tot = f_meta["fair_daily_quota_gb"] * horizon
        delta = proj_tot - fair_tot

        if delta > 0:
            st.warning(f"⚠️ **Projected Over-Consumption:** The next {horizon} days are forecast to consume **{proj_tot:.1f} GB**, which is **{delta:.1f} GB above** fair daily proportional allowances ({fair_tot:.1f} GB).")
        else:
            st.success(f"✅ **Safe Consumption:** The next {horizon} days are forecast at **{proj_tot:.1f} GB**, well within the fair quota buffer of {fair_tot:.1f} GB.")
