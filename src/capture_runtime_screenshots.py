"""
Capture High-Fidelity Live Runtime Screenshots using Selenium WebDriver
=======================================================================
Captures authentic, fully-rendered UI screenshots of all Streamlit pages:
1. dashboard_runtime.png        - Executive KPIs, Plotly Forecast Bridge, Quota Bar, Insights
2. record_usage_runtime.png     - Manual Data Logging Form, Validation, Device Breakdown
3. usage_forecast_runtime.png   - 7-Day Recursive Forecast Schedule & Prediction Intervals
4. analytics_models_runtime.png - Historical Analysis & Feature Correlations
5. anomaly_detection_runtime.png - Isolation Forest & IQR Outlier Diagnostics
6. model_benchmarks_runtime.png  - Model Leaderboard, RMSE/R2 Comparison & Feature Importance
"""

import os
import sys
import time
import subprocess
import urllib.request

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREENSHOTS_DIR = os.path.join(ROOT_DIR, "assets", "screenshots")

def is_server_running(url="http://localhost:8501/"):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

def wait_for_page_ready(driver, min_wait=5.0):
    """Wait until Streamlit finishes rendering charts and DOM elements."""
    try:
        WebDriverWait(driver, 15).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, "[data-testid='stAppViewContainer']")) > 0
        )
    except Exception:
        pass
    time.sleep(min_wait)

def capture_runtime_screenshots():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    
    server_process = None
    if not is_server_running():
        print("[1/6] Launching local Streamlit server on http://localhost:8501 ...")
        server_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        for i in range(20):
            time.sleep(1)
            if is_server_running():
                print(f"      Streamlit server is ready after {i+1}s!")
                break
        else:
            print("Error: Streamlit failed to start within 20s.")
            if server_process:
                server_process.kill()
            sys.exit(1)
    else:
        print("[1/6] Existing Streamlit instance detected on port 8501.")
        
    print("[2/6] Initializing headless Chrome WebDriver...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1400,1080")
    chrome_options.add_argument("--hide-scrollbars")
    chrome_options.add_argument("--log-level=3")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Page 1: Dashboard
        print("[3/6] Capturing 1: Executive Dashboard ...")
        driver.get("http://localhost:8501/")
        wait_for_page_ready(driver, min_wait=6.0)
        dash_path = os.path.join(SCREENSHOTS_DIR, "dashboard_runtime.png")
        driver.save_screenshot(dash_path)
        print(f"      [✓] Saved: {dash_path} ({os.path.getsize(dash_path):,} bytes)")
        
        # Page 2: Record Daily Usage
        print("[4/6] Capturing 2: Record Daily Usage Form ...")
        driver.get("http://localhost:8501/render_record_usage")
        wait_for_page_ready(driver, min_wait=5.0)
        rec_path = os.path.join(SCREENSHOTS_DIR, "record_usage_runtime.png")
        driver.save_screenshot(rec_path)
        print(f"      [✓] Saved: {rec_path} ({os.path.getsize(rec_path):,} bytes)")
        
        # Page 3: Usage Forecast
        print("[5/6] Capturing 3: Usage Forecast Engine ...")
        driver.get("http://localhost:8501/render_forecast")
        wait_for_page_ready(driver, min_wait=6.0)
        fore_path = os.path.join(SCREENSHOTS_DIR, "usage_forecast_runtime.png")
        driver.save_screenshot(fore_path)
        print(f"      [✓] Saved: {fore_path} ({os.path.getsize(fore_path):,} bytes)")
        
        # Page 4: Analytics & Models (Tab 1: EDA)
        print("[6/6] Capturing 4: Analytics & Models (Tab 1: Historical EDA) ...")
        driver.get("http://localhost:8501/render_analytics_models")
        wait_for_page_ready(driver, min_wait=6.0)
        an_path = os.path.join(SCREENSHOTS_DIR, "analytics_models_runtime.png")
        driver.save_screenshot(an_path)
        print(f"      [✓] Saved: {an_path} ({os.path.getsize(an_path):,} bytes)")
        
        # Click Tab 2: Anomaly Detection
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        if len(tabs) >= 2:
            print("      Switching to Tab 2: Anomaly Detection ...")
            driver.execute_script("arguments[0].click();", tabs[1])
            time.sleep(4.0)
            anom_path = os.path.join(SCREENSHOTS_DIR, "anomaly_detection_runtime.png")
            driver.save_screenshot(anom_path)
            print(f"      [✓] Saved: {anom_path} ({os.path.getsize(anom_path):,} bytes)")
            
        # Click Tab 3: Model Performance & Benchmarks
        if len(tabs) >= 3:
            print("      Switching to Tab 3: Model Performance & Benchmarks ...")
            driver.execute_script("arguments[0].click();", tabs[2])
            time.sleep(4.0)
            bench_path = os.path.join(SCREENSHOTS_DIR, "model_benchmarks_runtime.png")
            driver.save_screenshot(bench_path)
            print(f"      [✓] Saved: {bench_path} ({os.path.getsize(bench_path):,} bytes)")
            
        print("All runtime screenshots successfully captured with full visual fidelity!")
        
    finally:
        driver.quit()
        if server_process:
            print("Stopping background Streamlit server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except Exception:
                server_process.kill()

if __name__ == "__main__":
    capture_runtime_screenshots()
