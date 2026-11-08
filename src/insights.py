"""
Automated Narrative Insights Engine.
Extracts empirical patterns directly from user internet usage records
and synthesizes actionable household intelligence.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

def generate_household_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Examines the dataset and outputs a curated list of data-backed insights.
    Every insight is tied to actual computed metrics from the data.
    """
    insights = []
    
    if df.empty or len(df) < 7:
        return [{"type": "info", "title": "Insufficient Data", "body": "Need at least 7 days of records to generate reliable insights."}]
        
    df_sorted = df.copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date")
    
    # 1. Activity Dominance Analysis
    activity_cols = {
        "Video Streaming": "video_streaming_hours",
        "Social Media": "social_media_hours",
        "Online Gaming": "online_gaming_hours",
        "Web Browsing": "browsing_hours",
        "Video Calling": "video_calling_hours",
        "Online Learning": "online_learning_hours"
    }
    
    mean_hours = {}
    for label, col in activity_cols.items():
        if col in df_sorted.columns:
            mean_hours[label] = df_sorted[col].mean()
            
    if mean_hours:
        top_activity = max(mean_hours.items(), key=lambda x: x[1])
        insights.append({
            "type": "activity",
            "title": f"Primary Consumption Driver: {top_activity[0]}",
            "body": f"Household members spend an average of {top_activity[1]:.1f} hours/day on {top_activity[0].lower()}, making it the leading everyday online activity."
        })
        
    # 2. Weekend vs Weekday Disparity
    if "is_weekend" in df_sorted.columns and "total_data_usage_gb" in df_sorted.columns:
        weekend_mean = df_sorted[df_sorted["is_weekend"] == 1]["total_data_usage_gb"].mean()
        weekday_mean = df_sorted[df_sorted["is_weekend"] == 0]["total_data_usage_gb"].mean()
        
        if weekend_mean > weekday_mean:
            pct_diff = ((weekend_mean - weekday_mean) / weekday_mean) * 100
            insights.append({
                "type": "temporal",
                "title": "Elevated Weekend Traffic",
                "body": f"Weekend usage averages {weekend_mean:.1f} GB/day ({pct_diff:.1f}% higher than weekday average of {weekday_mean:.1f} GB/day), driven by recreational streaming and gaming."
            })
        else:
            pct_diff = ((weekday_mean - weekend_mean) / weekend_mean) * 100
            insights.append({
                "type": "temporal",
                "title": "Weekday Dominance (Work/Study Routine)",
                "body": f"Weekday usage averages {weekday_mean:.1f} GB/day ({pct_diff:.1f}% higher than weekend average of {weekend_mean:.1f} GB/day), indicating substantial remote work or virtual schooling demand."
            })
            
    # 3. Monthly Plan Limit Projection
    recent_month = df_sorted.tail(30)
    if "internet_plan_limit_gb" in df_sorted.columns:
        plan_limit = float(df_sorted["internet_plan_limit_gb"].iloc[-1])
        recent_daily_avg = float(recent_month["total_data_usage_gb"].mean())
        projected_monthly = recent_daily_avg * 30.0
        
        if projected_monthly > plan_limit:
            overage = projected_monthly - plan_limit
            insights.append({
                "type": "warning",
                "title": "High Risk of Data Cap Overage",
                "body": f"At recent consumption rates ({recent_daily_avg:.1f} GB/day), projected monthly usage is {projected_monthly:.1f} GB, exceeding the {plan_limit:.0f} GB plan limit by ~{overage:.1f} GB."
            })
        else:
            buffer = plan_limit - projected_monthly
            insights.append({
                "type": "success",
                "title": "Within Healthy Quota Limits",
                "body": f"Projected monthly consumption is {projected_monthly:.1f} GB ({round(projected_monthly/plan_limit*100, 1)}% of your {plan_limit:.0f} GB plan), leaving a comfortable {buffer:.1f} GB buffer."
            })
            
    # 4. Device Attribution
    device_cols = {
        "Smart TV": "smart_tv_usage_gb",
        "Laptop": "laptop_usage_gb",
        "Mobile": "mobile_usage_gb",
        "Tablet": "tablet_usage_gb"
    }
    dev_sums = {}
    for d_name, col in device_cols.items():
        if col in df_sorted.columns:
            dev_sums[d_name] = df_sorted[col].sum()
            
    if dev_sums:
        total_dev_usage = sum(dev_sums.values())
        top_dev = max(dev_sums.items(), key=lambda x: x[1])
        top_pct = (top_dev[1] / max(1.0, total_dev_usage)) * 100
        insights.append({
            "type": "device",
            "title": f"Leading Bandwidth Consumer: {top_dev[0]}",
            "body": f"{top_dev[0]} accounts for approximately {top_pct:.1f}% of total device traffic ({top_dev[1]:.0f} GB cumulative)."
        })
        
    # 5. Recent Volatility / Trend
    last_7_mean = df_sorted.tail(7)["total_data_usage_gb"].mean()
    prev_7_mean = df_sorted.tail(14).head(7)["total_data_usage_gb"].mean() if len(df_sorted) >= 14 else last_7_mean
    if prev_7_mean > 0:
        vol_change = ((last_7_mean - prev_7_mean) / prev_7_mean) * 100
        if abs(vol_change) > 12.0:
            direction = "surged by" if vol_change > 0 else "declined by"
            insights.append({
                "type": "trend",
                "title": f"Recent 7-Day Velocity Shift ({'+' if vol_change > 0 else ''}{vol_change:.1f}%)",
                "body": f"Average daily consumption over the past week ({last_7_mean:.1f} GB/day) {direction} {abs(vol_change):.1f}% relative to the preceding week ({prev_7_mean:.1f} GB/day)."
            })
            
    return insights
