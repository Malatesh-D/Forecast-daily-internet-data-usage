"""
Record Daily Usage View - NetUsage Analytics
Clean white-theme data entry, validation, edit/delete, and CSV batch operations.
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from src.database import (
    add_record,
    update_record,
    delete_record,
    load_combined_data,
    import_records_from_csv,
    calculate_date_features
)

def render_record_usage():
    st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #0f172a; }
        .main-header { font-size: 2.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1rem; color: #64748b; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">📝 Record Daily Internet Usage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Manually log daily internet usage, edit records, or perform batch CSV import and export.</div>', unsafe_allow_html=True)

    df_current = load_combined_data()
    latest_date_in_db = pd.to_datetime(df_current["date"]).max().date() if not df_current.empty else date.today()
    suggested_date = latest_date_in_db + timedelta(days=1)

    tab1, tab2, tab3 = st.tabs(["➕ New Entry", "✏️ Manage & Edit", "📤 Batch CSV Operations"])

    with tab1:
        st.markdown("#### 📋 Daily Usage Diary Entry")
        st.caption("Date features are automatically derived.")

        with st.form("entry_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                entry_date = st.date_input("Observation Date", value=suggested_date)
                d_str = entry_date.strftime("%Y-%m-%d")
                derived = calculate_date_features(d_str)
                st.info(f"📅 **{derived['day_of_week']}** | {'🏖️ Weekend' if derived['is_weekend'] else '🏢 Weekday'}")
            with c2:
                users = st.number_input("Household Users Active", min_value=1, max_value=10, value=3)
                devices = st.number_input("Connected Devices Count", min_value=1, max_value=30, value=8)
            with c3:
                plan_limit = st.number_input("Monthly Plan Limit (GB)", min_value=100.0, max_value=5000.0, value=750.0, step=50.0)
                wifi_hrs = st.number_input("Wi-Fi Connected Hours", min_value=0.0, max_value=24.0, value=16.0, step=0.5)

            st.markdown("##### 🎮 Core Activities (Hours & Downloads)")
            a1, a2, a3 = st.columns(3)
            with a1:
                v_stream = st.number_input("Video Streaming (hrs)", min_value=0.0, max_value=24.0, value=3.5, step=0.5)
                s_media = st.number_input("Social Media (hrs)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
            with a2:
                o_gaming = st.number_input("Online Gaming (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
                v_call = st.number_input("Video Calling (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
            with a3:
                downloads = st.number_input("Downloads (GB)", min_value=0.0, max_value=200.0, value=2.0, step=0.5)

            st.markdown("##### 📱 Device Consumption Breakdown (GB)")
            d1, d2, d3 = st.columns(3)
            with d1:
                tv_gb = st.number_input("Smart TV Usage (GB)", min_value=0.0, max_value=100.0, value=7.0, step=0.5)
            with d2:
                laptop_gb = st.number_input("Laptop Usage (GB)", min_value=0.0, max_value=100.0, value=4.5, step=0.5)
            with d3:
                mobile_gb = st.number_input("Mobile Usage (GB)", min_value=0.0, max_value=100.0, value=2.5, step=0.5)

            device_sum = tv_gb + laptop_gb + mobile_gb
            calc_total = round(device_sum + 1.2, 2)
            st.info(f"Estimated Total Usage: **{calc_total} GB** (Devices: {device_sum:.2f} GB + 1.2 GB IoT background)")

            submit = st.form_submit_button("💾 Save Record", use_container_width=True, type="primary")

        if submit:
            payload = {
                "date": d_str,
                "video_streaming_hours": v_stream,
                "social_media_hours": s_media,
                "online_gaming_hours": o_gaming,
                "browsing_hours": 2.0,
                "video_calling_hours": v_call,
                "downloading_gb": downloads,
                "online_learning_hours": 0.5,
                "mobile_usage_gb": mobile_gb,
                "laptop_usage_gb": laptop_gb,
                "smart_tv_usage_gb": tv_gb,
                "tablet_usage_gb": 1.0,
                "connected_devices_count": devices,
                "internet_plan_limit_gb": plan_limit,
                "network_speed_mbps": 150,
                "wifi_connected_hours": wifi_hrs,
                "number_of_users": users,
                "total_data_usage_gb": calc_total
            }
            ok, msg = add_record(payload)
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"⚠️ {msg}")

    with tab2:
        st.markdown("#### ✏️ View, Modify or Delete Records")
        all_dates = list(df_current["date"].values)[::-1]
        if all_dates:
            sel_date = st.selectbox("Select Date to View or Modify", options=all_dates)
            matches = df_current[df_current["date"] == sel_date]
            if not matches.empty:
                cur = matches.iloc[0].to_dict()
                with st.form("edit_form"):
                    e1, e2, e3 = st.columns(3)
                    with e1:
                        e_stream = st.number_input("Streaming (hrs)", value=float(cur.get("video_streaming_hours", 0.0) or 0.0), min_value=0.0, max_value=24.0)
                        e_social = st.number_input("Social Media (hrs)", value=float(cur.get("social_media_hours", 0.0) or 0.0), min_value=0.0, max_value=24.0)
                    with e2:
                        e_gaming = st.number_input("Gaming (hrs)", value=float(cur.get("online_gaming_hours", 0.0) or 0.0), min_value=0.0, max_value=24.0)
                        e_dl = st.number_input("Downloads (GB)", value=float(cur.get("downloading_gb", 0.0) or 0.0), min_value=0.0, max_value=200.0)
                    with e3:
                        e_tot = st.number_input("Total Usage (GB)", value=float(cur.get("total_data_usage_gb", 15.0) or 15.0), min_value=0.1, max_value=500.0)

                    b1, b2 = st.columns(2)
                    with b1:
                        up_btn = st.form_submit_button("🔄 Update Record", use_container_width=True, type="primary")
                    with b2:
                        del_btn = st.form_submit_button("🗑️ Delete Record", use_container_width=True)

                if up_btn:
                    cur.update({
                        "video_streaming_hours": e_stream,
                        "social_media_hours": e_social,
                        "online_gaming_hours": e_gaming,
                        "downloading_gb": e_dl,
                        "total_data_usage_gb": e_tot
                    })
                    ok, msg = update_record(sel_date, cur)
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

                if del_btn:
                    ok, msg = delete_record(sel_date)
                    if ok:
                        st.warning(f"⚠️ {msg}")
                    else:
                        st.error(f"❌ {msg}")

    with tab3:
        st.markdown("#### 📤 Batch CSV Data Operations")
        c_csv1, c_csv2 = st.columns(2)
        with c_csv1:
            st.markdown("##### 📥 Upload CSV")
            up_file = st.file_uploader("Upload usage records CSV", type=["csv"])
            if up_file is not None and st.button("🚀 Process & Import", use_container_width=True):
                imp, upd, msg = import_records_from_csv(up_file)
                st.success(msg)

        with c_csv2:
            st.markdown("##### 📤 Export Full Dataset")
            csv_bytes = df_current.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Full Dataset (CSV)",
                data=csv_bytes,
                file_name=f"internet_usage_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.info(f"Total dataset volume: **{len(df_current):,} records**.")
