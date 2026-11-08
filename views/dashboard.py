"""
Dashboard View - NetUsage Analytics
Crisp, clean white theme executive overview with KPIs, historical trends,
7-day forecast bridge, and dynamic narrative insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.database import load_combined_data
from src.forecasting import generate_recursive_forecast
from src.insights import generate_household_insights

def render_dashboard():
    # Clean White Theme CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #0f172a;
        }
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .white-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.1rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            margin-bottom: 0.5rem;
        }
        .kpi-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.2rem;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #059669;
            margin-top: 0.2rem;
            font-weight: 500;
        }
        .insight-pill {
            background-color: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 0.85rem 1.1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.75rem;
            border-top: 1px solid #f1f5f9;
            border-right: 1px solid #f1f5f9;
            border-bottom: 1px solid #f1f5f9;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="main-header">🌐 NetUsage Analytics Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Executive daily internet monitoring, device footprint, and 7-day predictive analytics.</div>', unsafe_allow_html=True)
    with col_h2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Load Data
    try:
        df = load_combined_data()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return

    if df.empty:
        st.warning("No records found. Please add usage records in the 'Record Daily Usage' tab.")
        return

    df["date_dt"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values("date_dt").reset_index(drop=True)

    latest = df_sorted.iloc[-1]
    latest_date_str = str(latest["date"])
    total_records = len(df_sorted)

    # 7-day forecast
    forecast_df, f_meta = generate_recursive_forecast(df_sorted, horizon_days=7)
    tomorrow_pred = forecast_df.iloc[0]["predicted_usage_gb"]

    # Metrics
    today_usage = float(latest["total_data_usage_gb"])
    avg_daily_usage = float(df_sorted["total_data_usage_gb"].mean())
    max_usage_val = float(df_sorted["total_data_usage_gb"].max())

    latest_dt = latest["date_dt"]
    curr_month = df_sorted[(df_sorted["date_dt"].dt.year == latest_dt.year) & (df_sorted["date_dt"].dt.month == latest_dt.month)]
    month_to_date = float(curr_month["total_data_usage_gb"].sum())
    plan_limit = float(latest.get("internet_plan_limit_gb", 750.0))
    remaining_quota = max(0.0, plan_limit - month_to_date)
    pct_used = min(100.0, (month_to_date / plan_limit) * 100)

    # 1. KPI Cards Row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="white-card">
            <div class="kpi-title">Today's Usage</div>
            <div class="kpi-value">{today_usage:.1f} <span style="font-size:0.9rem;color:#64748b;">GB</span></div>
            <div class="kpi-sub">📅 {latest_date_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="white-card">
            <div class="kpi-title">Tomorrow's Forecast</div>
            <div class="kpi-value" style="color:#2563eb;">{tomorrow_pred:.1f} <span style="font-size:0.9rem;color:#64748b;">GB</span></div>
            <div class="kpi-sub">🤖 Best Model</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="white-card">
            <div class="kpi-title">Daily Average</div>
            <div class="kpi-value">{avg_daily_usage:.1f} <span style="font-size:0.9rem;color:#64748b;">GB</span></div>
            <div class="kpi-sub">Over {total_records} days</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="white-card">
            <div class="kpi-title">Peak Day</div>
            <div class="kpi-value" style="color:#dc2626;">{max_usage_val:.1f} <span style="font-size:0.9rem;color:#64748b;">GB</span></div>
            <div class="kpi-sub">Historical max</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="white-card">
            <div class="kpi-title">Month-To-Date</div>
            <div class="kpi-value">{month_to_date:.1f} <span style="font-size:0.9rem;color:#64748b;">GB</span></div>
            <div class="kpi-sub">{pct_used:.1f}% consumed</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        q_color = "#059669" if remaining_quota > 100 else "#dc2626"
        st.markdown(f"""
        <div class="white-card">
            <div class="kpi-title">Remaining Quota</div>
            <div class="kpi-value" style="color:{q_color};">{remaining_quota:.1f} <span style="font-size:0.9rem;color:#64748b;">GB</span></div>
            <div class="kpi-sub">Plan: {plan_limit:.0f} GB</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 2. Main Timeline Chart & Quota Card
    c_chart1, c_chart2 = st.columns([2, 1])

    with c_chart1:
        st.markdown("#### 📈 Historical Usage with 7-Day Forward Forecast")
        recent_hist = df_sorted.tail(60).copy()

        fig_main = go.Figure()

        # Actual Line
        fig_main.add_trace(go.Scatter(
            x=recent_hist["date_dt"],
            y=recent_hist["total_data_usage_gb"],
            mode="lines+markers",
            name="Actual (Past 60 Days)",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=4)
        ))

        # Forecast Line
        f_dates = pd.to_datetime(forecast_df["date"])
        bridge_x = [recent_hist["date_dt"].iloc[-1]] + list(f_dates)
        bridge_y = [recent_hist["total_data_usage_gb"].iloc[-1]] + list(forecast_df["predicted_usage_gb"])

        fig_main.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="7-Day Forecast",
            line=dict(color="#10b981", width=2.5, dash="dash"),
            marker=dict(size=6, color="#059669")
        ))

        # Prediction Interval
        fig_main.add_trace(go.Scatter(
            x=list(f_dates) + list(f_dates)[::-1],
            y=list(forecast_df["upper_interval_90_gb"]) + list(forecast_df["lower_interval_90_gb"])[::-1],
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="90% Prediction Interval"
        ))

        fig_main.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Usage (GB)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_main, use_container_width=True)

    with c_chart2:
        st.markdown("#### 📊 Quota & Device Footprint")
        st.caption(f"Monthly Plan Usage ({month_to_date:.1f} / {plan_limit:.0f} GB)")
        st.progress(min(1.0, month_to_date / plan_limit))

        # Device Donut
        dev_cols = ["smart_tv_usage_gb", "laptop_usage_gb", "mobile_usage_gb"]
        avail_dev = [c for c in dev_cols if c in df_sorted.columns]
        if avail_dev:
            recent_30 = df_sorted.tail(30)
            dev_sum = recent_30[avail_dev].sum()
            labels = ["Smart TV", "Laptop", "Mobile Phone"]

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels,
                values=dev_sum.values,
                hole=.6,
                marker_colors=["#2563eb", "#10b981", "#f59e0b"]
            )])
            fig_donut.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                height=250,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # 3. Dynamic Narrative Insights
    st.markdown("### 💡 Data-Driven Household Insights")
    insights = generate_household_insights(df_sorted)

    col_ins1, col_ins2 = st.columns(2)
    for i, item in enumerate(insights):
        target = col_ins1 if i % 2 == 0 else col_ins2
        border_c = "#2563eb"
        if item.get("type") == "warning":
            border_c = "#ef4444"
        elif item.get("type") == "success":
            border_c = "#10b981"
        elif item.get("type") == "temporal":
            border_c = "#8b5cf6"

        with target:
            st.markdown(f"""
            <div class="insight-pill" style="border-left-color: {border_c};">
                <strong style="color:#0f172a;">{item['title']}</strong><br/>
                <span style="font-size:0.88rem; color:#475569;">{item['body']}</span>
            </div>
            """, unsafe_allow_html=True)
