"""
Database and Data Management Module.
Provides SQLite persistent storage for manually recorded usage data,
duplicate detection, record editing/deleting, CSV import/export,
and unified access combining historical baseline and live entries.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

DB_PATH = os.path.join("data", "usage_tracker.db")
BASE_CSV_PATH = os.path.join("data", "internet_usage.csv")
MANUAL_CSV_PATH = os.path.join("data", "manual_usage.csv")

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create and return a database connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_PATH):
    """Initialize the SQLite database schema if not already present."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                date TEXT PRIMARY KEY,
                day_of_week TEXT,
                is_weekend INTEGER,
                month INTEGER,
                season TEXT,
                is_holiday INTEGER,
                video_streaming_hours REAL,
                social_media_hours REAL,
                online_gaming_hours REAL,
                browsing_hours REAL,
                video_calling_hours REAL,
                downloading_gb REAL,
                online_learning_hours REAL,
                mobile_usage_gb REAL,
                laptop_usage_gb REAL,
                smart_tv_usage_gb REAL,
                tablet_usage_gb REAL,
                connected_devices_count INTEGER,
                internet_plan_limit_gb REAL,
                network_speed_mbps INTEGER,
                wifi_connected_hours REAL,
                number_of_users INTEGER,
                total_data_usage_gb REAL,
                is_manual INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

def calculate_date_features(date_str: str) -> Dict[str, Any]:
    """Derive calendar and seasonal features from a date string (YYYY-MM-DD)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    month = dt.month
    day_of_week = dt.strftime("%A")
    is_weekend = 1 if dt.weekday() >= 5 else 0
    
    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Autumn"
        
    is_holiday = 0
    if (month == 1 and dt.day in [1, 2]) or \
       (month == 7 and dt.day in [4]) or \
       (month == 12 and dt.day in [24, 25, 26, 31]) or \
       (month == 11 and 22 <= dt.day <= 28 and dt.weekday() == 3) or \
       (month == 5 and 24 <= dt.day <= 31 and dt.weekday() == 0):
        is_holiday = 1
        
    return {
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "month": month,
        "season": season,
        "is_holiday": is_holiday
    }

def record_exists(date_str: str, db_path: str = DB_PATH, check_base_csv: bool = True) -> bool:
    """Check if a record for the specified date already exists in SQLite or base CSV."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM daily_usage WHERE date = ?", (date_str,))
        if cursor.fetchone() is not None:
            return True
    finally:
        conn.close()
            
    if check_base_csv and os.path.exists(BASE_CSV_PATH):
        try:
            df = pd.read_csv(BASE_CSV_PATH, usecols=["date"])
            if date_str in df["date"].values:
                return True
        except Exception:
            pass
            
    return False

def add_record(record_data: Dict[str, Any], db_path: str = DB_PATH) -> Tuple[bool, str]:
    """
    Insert a new usage record into SQLite and sync with manual_usage.csv.
    """
    init_db(db_path)
    date_str = str(record_data["date"])
    
    if record_exists(date_str, db_path, check_base_csv=True):
        return False, f"Record for date {date_str} already exists. Please use edit functionality."
        
    # Enrich with date features if not present
    derived = calculate_date_features(date_str)
    for k, v in derived.items():
        if k not in record_data or record_data[k] is None:
            record_data[k] = v
            
    fields = [
        "date", "day_of_week", "is_weekend", "month", "season", "is_holiday",
        "video_streaming_hours", "social_media_hours", "online_gaming_hours",
        "browsing_hours", "video_calling_hours", "downloading_gb", "online_learning_hours",
        "mobile_usage_gb", "laptop_usage_gb", "smart_tv_usage_gb", "tablet_usage_gb",
        "connected_devices_count", "internet_plan_limit_gb", "network_speed_mbps",
        "wifi_connected_hours", "number_of_users", "total_data_usage_gb", "is_manual"
    ]
    
    record_data["is_manual"] = 1
    values = [record_data.get(f) for f in fields]
    placeholders = ",".join(["?"] * len(fields))
    col_str = ",".join(fields)
    
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO daily_usage ({col_str}) VALUES ({placeholders})", values)
        conn.commit()
    except Exception as e:
        return False, f"Failed to add record: {str(e)}"
    finally:
        conn.close()
        
    sync_manual_csv(db_path)
    return True, f"Successfully recorded usage data for {date_str}."

def update_record(date_str: str, record_data: Dict[str, Any], db_path: str = DB_PATH) -> Tuple[bool, str]:
    """Update an existing record in SQLite."""
    init_db(db_path)
    
    derived = calculate_date_features(date_str)
    for k, v in derived.items():
        record_data[k] = v
        
    fields = [
        "day_of_week", "is_weekend", "month", "season", "is_holiday",
        "video_streaming_hours", "social_media_hours", "online_gaming_hours",
        "browsing_hours", "video_calling_hours", "downloading_gb", "online_learning_hours",
        "mobile_usage_gb", "laptop_usage_gb", "smart_tv_usage_gb", "tablet_usage_gb",
        "connected_devices_count", "internet_plan_limit_gb", "network_speed_mbps",
        "wifi_connected_hours", "number_of_users", "total_data_usage_gb"
    ]
    
    set_clause = ", ".join([f"{f} = ?" for f in fields])
    values = [record_data.get(f) for f in fields]
    values.append(date_str)
    
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE daily_usage SET {set_clause} WHERE date = ?", values)
        if cursor.rowcount == 0:
            # If not in SQLite yet (maybe was in base CSV), insert as modified record in SQLite
            record_data["date"] = date_str
            record_data["is_manual"] = 1
            return add_record(record_data, db_path)
        conn.commit()
    except Exception as e:
        return False, f"Failed to update record: {str(e)}"
    finally:
        conn.close()
        
    sync_manual_csv(db_path)
    return True, f"Successfully updated record for {date_str}."

def delete_record(date_str: str, db_path: str = DB_PATH) -> Tuple[bool, str]:
    """Delete a record by date from SQLite."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM daily_usage WHERE date = ?", (date_str,))
        conn.commit()
    except Exception as e:
        return False, f"Failed to delete record: {str(e)}"
    finally:
        conn.close()
        
    sync_manual_csv(db_path)
    return True, f"Record for {date_str} deleted from manual database."

def sync_manual_csv(db_path: str = DB_PATH, csv_path: str = MANUAL_CSV_PATH):
    """Write all SQLite records to manual_usage.csv for backup and portability."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM daily_usage ORDER BY date ASC", conn)
    finally:
        conn.close()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

def load_combined_data(db_path: str = DB_PATH, base_csv_path: str = BASE_CSV_PATH) -> pd.DataFrame:
    """
    Load unified dataset combining synthetic historical data and live SQLite manual records.
    Manual records take precedence in case of identical dates.
    Ensures chronological ordering.
    """
    init_db(db_path)
    frames = []
    
    # 1. Base synthetic CSV
    if os.path.exists(base_csv_path):
        base_df = pd.read_csv(base_csv_path)
        if "is_manual" not in base_df.columns:
            base_df["is_manual"] = 0
        frames.append(base_df)
        
    # 2. SQLite manual records
    conn = get_connection(db_path)
    try:
        manual_df = pd.read_sql_query("SELECT * FROM daily_usage", conn)
        if not manual_df.empty:
            if "created_at" in manual_df.columns:
                manual_df = manual_df.drop(columns=["created_at"])
            frames.append(manual_df)
    finally:
        conn.close()
            
    if not frames:
        return pd.DataFrame()
        
    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate keeping manual entries (last ones)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    return combined

def import_records_from_csv(file_obj_or_path, db_path: str = DB_PATH) -> Tuple[int, int, str]:
    """Import multiple usage records from an uploaded CSV into SQLite."""
    init_db(db_path)
    try:
        df = pd.read_csv(file_obj_or_path)
    except Exception as e:
        return 0, 0, f"Error reading CSV: {str(e)}"
        
    if "date" not in df.columns:
        return 0, 0, "Uploaded CSV must contain a 'date' column in YYYY-MM-DD format."
        
    imported = 0
    skipped = 0
    
    for _, row in df.iterrows():
        rec = row.to_dict()
        d_str = str(rec["date"]).strip()
        if record_exists(d_str, db_path, check_base_csv=False):
            # Update existing
            update_record(d_str, rec, db_path)
            skipped += 1
        else:
            add_record(rec, db_path)
            imported += 1
            
    return imported, skipped, f"Import complete: {imported} new records added, {skipped} existing records updated."
