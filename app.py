"""
NetUsage Analytics - Streamlined Executive Application
Configured with modern st.navigation and crisp pure-white theme.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from views.dashboard import render_dashboard
from views.record_usage import render_record_usage
from views.forecast import render_forecast
from views.analytics_models import render_analytics_models

st.set_page_config(
    page_title="NetUsage Analytics | Daily Internet Forecasting",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Pure White CSS
st.markdown("""
<style>
    /* Force Pure White Background Across Application */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    /* Clean Light Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
    }
    
    /* Sidebar Navigation Links */
    [data-testid="stSidebarNavLink"] {
        color: #334155 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.75rem !important;
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stSidebarNavLink"]:hover {
        background-color: #edf2f7 !important;
        color: #1e3a8a !important;
    }
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background-color: #dbeafe !important;
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #0f172a !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #475569 !important;
        font-weight: 500;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1d4ed8 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Define 4 Streamlined Pages
dashboard_page = st.Page(render_dashboard, title="Dashboard", icon="📊", default=True)
record_page = st.Page(render_record_usage, title="Record Daily Usage", icon="📝")
forecast_page = st.Page(render_forecast, title="Usage Forecast", icon="🔮")
analytics_page = st.Page(render_analytics_models, title="Analytics & Models", icon="📈")

pg = st.navigation([dashboard_page, record_page, forecast_page, analytics_page])

# Sidebar Brand Header
with st.sidebar:
    st.markdown("### 🌐 **NetUsage Analytics**")
    st.caption("Predictive Analytics Mini Project")
    st.divider()

pg.run()
