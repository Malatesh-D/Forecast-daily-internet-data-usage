"""
Data Generator for Daily Internet Usage Forecasting System.
Generates realistic, academically defensible synthetic household diary data
spanning 2020 to 2026 (~2,200 days) with realistic seasonality, day-of-week patterns,
occasional holidays, household user variations, device breakdowns, and realistic noise/missing values.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_season(month: int) -> str:
    """Return meteorological season for a given month."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def generate_synthetic_data(num_days: int = 2200, start_date: str = "2020-01-01", random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic internet consumption diary records.
    
    Parameters:
    -----------
    num_days : int
        Number of daily records to generate (default 2200, > 6 years).
    start_date : str
        Initial date in YYYY-MM-DD.
    random_seed : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all required features and realistic consumption relationships.
    """
    np.random.seed(random_seed)
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    date_list = [base_date + timedelta(days=i) for i in range(num_days)]
    
    records = []
    
    # Household baseline profile parameters
    # Let's model a realistic 3-4 person household with slight growth in internet dependence over 2020-2026
    for i, current_date in enumerate(date_list):
        day_of_week = current_date.strftime("%A")
        is_weekend = 1 if current_date.weekday() >= 5 else 0
        month = current_date.month
        season = get_season(month)
        
        # Realistic holiday indicator: New Year, Christmas, July 4, Thanksgiving, Labour Day, Memorial Day, etc.
        # Simple rule-based holiday check + ~12 fixed dates per year
        is_holiday = 0
        if (month == 1 and current_date.day in [1, 2]) or \
           (month == 7 and current_date.day in [4]) or \
           (month == 12 and current_date.day in [24, 25, 26, 31]) or \
           (month == 11 and 22 <= current_date.day <= 28 and current_date.weekday() == 3) or \
           (month == 5 and 24 <= current_date.day <= 31 and current_date.weekday() == 0):
            is_holiday = 1
        
        # Household baseline trends:
        # Slight long-term growth trend (2020 to 2026: internet usage gradually increases as 4K and richer apps grow)
        trend_factor = 1.0 + (i / num_days) * 0.35  # ~35% growth over 6 years
        
        # Weekend & Holiday boost factor
        leisure_multiplier = 1.0
        if is_weekend:
            leisure_multiplier += np.random.uniform(0.35, 0.65)
        if is_holiday:
            leisure_multiplier += np.random.uniform(0.40, 0.85)
            
        # Season factor (Summer vacation and Winter holiday binge watching)
        season_mult = 1.0
        if season == "Winter":
            season_mult = 1.15
        elif season == "Summer":
            season_mult = 1.10
        elif season == "Spring":
            season_mult = 0.95
        else:
            season_mult = 1.0
            
        # Occasional household guests or vacation days
        # 1% chance of vacation/away from home (very low usage)
        # 3% chance of high-consumption party/movie marathon/game release
        is_away = np.random.rand() < 0.015
        is_spike_day = (not is_away) and (np.random.rand() < 0.04)
        
        if is_away:
            # Low usage day
            num_users = 1
            connected_devices = np.random.randint(2, 5)
            wifi_hours = np.random.uniform(4.0, 8.0)
            video_streaming_hrs = np.random.uniform(0.0, 0.8)
            social_media_hrs = np.random.uniform(0.2, 1.0)
            online_gaming_hrs = 0.0
            browsing_hrs = np.random.uniform(0.2, 0.8)
            video_calling_hrs = 0.0
            downloading_gb = 0.0
            online_learning_hrs = 0.0
        else:
            # Regular day with dynamic activity
            # Household users: 2 to 4 people, occasionally 5
            num_users = int(np.clip(np.random.normal(3.2, 0.7), 2, 5))
            connected_devices = int(np.clip(num_users * 2 + np.random.randint(1, 4), 4, 14))
            wifi_hours = float(np.clip(np.random.normal(16.5, 3.0), 8.0, 24.0))
            
            # Activities
            base_streaming = (1.8 * num_users * 0.6) * leisure_multiplier * season_mult * (1.3 if is_spike_day else 1.0)
            video_streaming_hrs = float(np.clip(np.random.normal(base_streaming, 0.8), 0.5, 12.0))
            
            base_social = (1.2 * num_users * 0.5) * (1.1 if is_weekend else 0.95)
            social_media_hrs = float(np.clip(np.random.normal(base_social, 0.5), 0.3, 8.0))
            
            # Gaming is higher on weekends
            base_gaming = (0.8 if is_weekend else 0.3) * (num_users * 0.4) * (2.2 if is_spike_day else 1.0)
            online_gaming_hrs = float(np.clip(np.random.exponential(base_gaming), 0.0, 7.0))
            
            # Browsing
            browsing_hrs = float(np.clip(np.random.normal(2.5, 0.7), 0.5, 6.0))
            
            # Video calling (higher on weekdays for work/school, or weekend family catch-ups)
            base_calling = 1.2 if not is_weekend else 0.8
            video_calling_hrs = float(np.clip(np.random.normal(base_calling, 0.6), 0.0, 5.0))
            
            # Online learning (school/weekday dominant)
            base_learning = (1.5 if (not is_weekend and season in ["Spring", "Autumn", "Winter"]) else 0.2)
            online_learning_hrs = float(np.clip(np.random.normal(base_learning, 0.5), 0.0, 4.5))
            
            # Downloading GB (games, patches, software updates, torrents, OS updates)
            if is_spike_day:
                downloading_gb = float(np.random.uniform(15.0, 45.0))
            else:
                downloading_gb = float(np.random.choice(
                    [np.random.uniform(0.1, 2.0), np.random.uniform(2.0, 8.0)],
                    p=[0.78, 0.22]
                ))

        # Rates of data consumption per activity hour (realistic estimates):
        # Video streaming: ~2.2 GB/hr (mix of 1080p and 4K Netflix/YouTube)
        # Social media: ~0.35 GB/hr (Instagram reels, TikTok, Twitter media)
        # Online gaming: ~0.15 GB/hr (packet data; game downloads handled separately)
        # Browsing: ~0.12 GB/hr (articles, search, web pages)
        # Video calling: ~1.1 GB/hr (Zoom, Google Meet, Teams, FaceTime 720p/1080p)
        # Online learning: ~0.65 GB/hr (video lectures, canvas, LMS interactive content)
        
        activity_usage_est = (
            video_streaming_hrs * 2.2 +
            social_media_hrs * 0.35 +
            online_gaming_hrs * 0.15 +
            browsing_hrs * 0.12 +
            video_calling_hrs * 1.1 +
            online_learning_hrs * 0.65 +
            downloading_gb
        ) * trend_factor
        
        # Shared household background overhead (smart speakers, security cameras, cloud sync, background OS pings):
        background_overhead = np.random.uniform(0.4, 1.8) * (connected_devices / 6.0)
        
        # Calculate Total Data Usage
        # Add realistic observation diary measurement error (+/- 4%)
        total_data_usage_gb = float(np.round(activity_usage_est + background_overhead + np.random.normal(0, 0.45), 2))
        total_data_usage_gb = max(0.5, total_data_usage_gb)
        
        # Device usage breakdown - must be internally consistent with total consumption,
        # with small realistic variations due to background traffic / IoT / shared device overlap
        # Mobile: social media + some streaming + mobile browsing
        # Smart TV: predominantly video streaming
        # Laptop: work/learning, video calling, downloading, browsing, some streaming
        # Tablet: casual streaming, learning, social media
        
        tv_share_base = (video_streaming_hrs * 1.5) / max(0.1, activity_usage_est)
        laptop_share_base = (video_calling_hrs * 1.0 + online_learning_hrs * 0.5 + downloading_gb * 0.7 + browsing_hrs * 0.08) / max(0.1, activity_usage_est)
        mobile_share_base = (social_media_hrs * 0.3 + browsing_hrs * 0.04 + video_streaming_hrs * 0.4) / max(0.1, activity_usage_est)
        tablet_share_base = (online_learning_hrs * 0.15 + video_streaming_hrs * 0.3) / max(0.1, activity_usage_est)
        
        raw_shares = np.array([mobile_share_base, laptop_share_base, tv_share_base, tablet_share_base], dtype=float)
        if raw_shares.sum() <= 0:
            raw_shares = np.array([0.3, 0.3, 0.25, 0.15])
        shares = raw_shares / raw_shares.sum()
        
        # Allocate device usages
        device_total_pool = total_data_usage_gb - (background_overhead * 0.7) # IoT overhead not attributed to single personal device
        mobile_usage_gb = float(np.round(max(0.1, device_total_pool * shares[0] + np.random.normal(0, 0.15)), 2))
        laptop_usage_gb = float(np.round(max(0.1, device_total_pool * shares[1] + np.random.normal(0, 0.2)), 2))
        smart_tv_usage_gb = float(np.round(max(0.0, device_total_pool * shares[2] + np.random.normal(0, 0.2)), 2))
        tablet_usage_gb = float(np.round(max(0.0, device_total_pool * shares[3] + np.random.normal(0, 0.1)), 2))
        
        # Network & Subscription details
        # Plan limit: typical 500GB or 1000GB monthly plan (~16.6 to 33.3 GB daily allowance)
        internet_plan_limit_gb = 750.0  # standard monthly limit
        network_speed_mbps = int(np.random.choice([100, 150, 200, 300], p=[0.25, 0.40, 0.25, 0.10]))
        
        records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": month,
            "season": season,
            "is_holiday": is_holiday,
            "video_streaming_hours": round(video_streaming_hrs, 2),
            "social_media_hours": round(social_media_hrs, 2),
            "online_gaming_hours": round(online_gaming_hrs, 2),
            "browsing_hours": round(browsing_hrs, 2),
            "video_calling_hours": round(video_calling_hrs, 2),
            "downloading_gb": round(downloading_gb, 2),
            "online_learning_hours": round(online_learning_hrs, 2),
            "mobile_usage_gb": mobile_usage_gb,
            "laptop_usage_gb": laptop_usage_gb,
            "smart_tv_usage_gb": smart_tv_usage_gb,
            "tablet_usage_gb": tablet_usage_gb,
            "connected_devices_count": connected_devices,
            "internet_plan_limit_gb": internet_plan_limit_gb,
            "network_speed_mbps": network_speed_mbps,
            "wifi_connected_hours": round(wifi_hours, 2),
            "number_of_users": num_users,
            "total_data_usage_gb": total_data_usage_gb
        })
        
    df = pd.DataFrame(records)
    
    # Introduce a realistic ~1.5% sparse missing values in non-critical activity & device fields
    # to realistically demonstrate missing value treatment in academic evaluation.
    # Never introduce missing values in date or target!
    missing_cols = ["video_calling_hours", "tablet_usage_gb", "wifi_connected_hours", "online_learning_hours"]
    for col in missing_cols:
        mask = np.random.rand(len(df)) < 0.015
        df.loc[mask, col] = np.nan
        
    return df

def save_initial_dataset(output_path: str = "data/internet_usage.csv"):
    """Generate and save the synthetic historical dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_data(num_days=2200, start_date="2020-01-01")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at {output_path} with {len(df)} rows and {df.shape[1]} columns.")
    return df

if __name__ == "__main__":
    save_initial_dataset()
