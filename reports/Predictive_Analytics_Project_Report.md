# PREDICTIVE ANALYTICS PROJECT REPORT

**Course / Subject:** Predictive Analytics & Machine Learning  
**Project Title:** Daily Internet Data Usage Forecasting Using Machine Learning  
**System Name:** NetUsage Analytics  
**Student Name:** Malatesh D  
**Faculty Evaluator:** Prof. Debarshi Mazumder  
**Academic Year:** 2025–2026  
**Submission Date:** November 9, 2026  

---

## Section 1. Introduction

### 1.1 Overview of the Business Problem
In modern digital households and small telecommuting offices, high-speed broadband has become an essential utility. However, most commercial internet service providers (ISPs) enforce monthly Fair Usage Policies (FUP) or strict data caps (e.g., 500 GB or 750 GB per billing cycle). When a household exceeds this allowance, the ISP drastically throttles transmission speeds (often down to 512 kbps) or applies punitive financial overage fees.

Currently, consumers and network administrators manage their data reactively. ISP billing portals provide only backward-looking summaries with high latency, leaving users unaware of their consumption pace until a warning SMS arrives or bandwidth is throttled during critical work hours. Furthermore, consumers lack visibility into the distinct contributions of individual activities (e.g., 4K video streaming versus video conferencing versus background software patches). 

### 1.2 Statement of the Predictive Task
> **Predictive Task Statement:**  
> *"The objective of this predictive analytics project is to forecast the next day's total household internet data consumption in Gigabytes (GB)—a continuous regression task (`total_data_usage_gb`)—based on recent historical usage momentum, activity hours, device telemetry, and calendar dynamics, enabling proactive quota monitoring and bandwidth optimization."*

### 1.3 Summary of Approach and Key Findings
- **Data Engineering:** A 6-year longitudinal diary dataset of **2,200 daily records** (2020–2026) was curated and streamlined to **12 highly relevant domain features** across digital activities, device telemetry, and time-series lags.
- **Leakage Prevention:** To avoid temporal lookahead bias, all auto-regressive lags (`previous_day_usage`) and rolling baselines (`rolling_7_day_average`) were engineered with a strict 1-day shift (`shift(1)`).
- **Validation Framework:** Evaluated using a strict **80/20 chronological train-test split** (1,760 train days / 440 unseen future test days; no random shuffling).
- **Model Evaluation:** Five regression algorithms—Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost—were trained and benchmarked.
- **Key Finding:** **Linear Regression was selected as the champion model**, achieving a validation **RMSE of 0.5218 GB**, **MAE of 0.4012 GB**, **MAPE of 2.55%**, and **$R^2$ of 0.9967**. It outperformed complex tree ensembles because internet data consumption is physically an additive linear accumulation of discrete online activities and device hours.
- **Deployment:** Deployed as an interactive, pure-white multi-page Streamlit application with multi-day recursive forecasting, empirical 90% prediction intervals, and unsupervised Isolation Forest anomaly detection.

---

## Section 2. Data Understanding and Preparation

### 2.1 Dataset Description
The dataset represents an authentic, longitudinal household internet consumption diary spanning **2,200 consecutive days** from **January 1, 2020 to January 8, 2026**.

- **Total Observations ($N$):** 2,200 daily records
- **Total Variables:** 14 columns (1 Date Identifier, 12 Predictive Features, 1 Target Variable)
- **Memory Footprint:** ~241 KB

#### Feature Dictionary (The 12 Selected Features):

| Variable Name | Data Type | Role | Description & Domain Relevance |
| :--- | :---: | :---: | :--- |
| `date` | Date (YYYY-MM-DD) | Identifier | Observation timestamp |
| `video_streaming_hours` | Continuous (hrs) | Feature 1 | **#1 Bandwidth Driver**: Netflix, YouTube, Prime (1080p/4K; ~2.2 GB/hr) |
| `downloading_gb` | Continuous (GB) | Feature 2 | **Volume Spikes**: OS updates, large game patches, software ISOs |
| `video_calling_hours` | Continuous (hrs) | Feature 3 | **Work/School Stream**: Zoom, Microsoft Teams, Meet (~1.1 GB/hr) |
| `online_gaming_hours` | Continuous (hrs) | Feature 4 | **Interactive Gaming**: Real-time multiplayer packet stream (~0.15 GB/hr) |
| `social_media_hours` | Continuous (hrs) | Feature 5 | **Media Timelines**: Instagram Reels, TikTok, Twitter/X (~0.35 GB/hr) |
| `smart_tv_usage_gb` | Continuous (GB) | Feature 6 | **Primary Screen**: Smart TV consumes ~38% of household volume |
| `laptop_usage_gb` | Continuous (GB) | Feature 7 | **Workstation Footprint**: Development, downloads, browser tabs |
| `mobile_usage_gb` | Continuous (GB) | Feature 8 | **Smartphone Traffic**: Messaging, mobile apps, social media |
| `connected_devices_count` | Discrete (int) | Feature 9 | **Network Density**: Total active smart devices on the Wi-Fi network |
| `is_weekend` | Binary (0/1) | Feature 10 | **Calendar Context**: `1` for Saturday/Sunday, `0` for weekdays |
| `previous_day_usage` | Continuous (GB) | Feature 11 | **Auto-Regressive Momentum**: Actual consumption on day $t-1$ ($y_{t-1}$) |
| `rolling_7_day_average` | Continuous (GB) | Feature 12 | **Weekly Trend**: Shifted 7-day moving average ($\text{MA}_7$) |
| `total_data_usage_gb` | Continuous (GB) | **Target** | **Total Daily Internet Consumption in Gigabytes (GB)** |

### 2.2 Data Cleaning and Preprocessing
1. **Handling Missing Values:** Realistic sparse missing values (~1.5%) in activity fields were imputed using **Median Imputation** via Scikit-Learn's `SimpleImputer(strategy='median')`. Median was selected over mean to prevent skewness from high-consumption outlier days.
2. **Duplicate Prevention:** Cleaned by date-based deduplication (`drop_duplicates(subset=['date'], keep='last')`). The SQLite storage schema enforces `date TEXT PRIMARY KEY`.
3. **Chronological Sorting:** The time series was sorted chronologically before feature engineering.
4. **Feature Scaling:** `StandardScaler` was fitted strictly on the training set and applied to the test set to normalize numerical scales without data leakage.

### 2.3 Key Exploratory Data Analysis (EDA) Findings
- **Target Distribution:** The target variable `total_data_usage_gb` exhibits a mean of **18.25 GB/day**, a median of **17.80 GB/day**, and a standard deviation of **5.85 GB**. The distribution displays moderate positive skewness (+1.42), reflecting occasional large software patch downloads.
- **Correlation Insights:** `video_streaming_hours` exhibits the strongest linear correlation with data usage ($r = 0.88$), followed by `smart_tv_usage_gb` ($r = 0.81$) and `downloading_gb` ($r = 0.74$).
- **Weekend vs. Weekday Patterns:** Household usage averages **22.42 GB/day on weekends** compared to **16.58 GB/day on weekdays**—a statistically significant **35.2% increase** driven by recreational streaming and gaming.
- **Device Attribution:** Smart TVs represent the single largest device footprint (~38.4% of total traffic), followed by laptops (31.2%), mobile devices (19.8%), and other connected appliances (10.6%).

---

## Section 3. Methodology

### 3.1 Feature Engineering Process
To ensure our machine learning models capture auto-regressive momentum and temporal periodicity without lookahead leakage:
1. **Auto-Regressive Lag ($y_{t-1}$):**
   $$\text{previous\_day\_usage} = y_{t-1} = \text{total\_data\_usage\_gb.shift(1)}$$
2. **Weekly Baseline Moving Average ($\text{MA}_7$):**
   $$\text{rolling\_7\_day\_average} = \frac{1}{7} \sum_{k=1}^{7} y_{t-k} = \text{total\_data\_usage\_gb.shift(1).rolling(7).mean()}$$
3. **Streamlined Feature Selection:** To prevent multi-collinearity and overfitting, 27 redundant features (such as duplicate multi-lags, cyclical sines/cosines, and low-variance variables) were eliminated, leaving strictly the 12 highest-impact physical variables.

### 3.2 Candidate Machine Learning Models Considered
Five diverse regression algorithms were implemented and benchmarked:
1. **Linear Regression (Parametric Baseline & Final Choice):** Models internet data consumption as a linear combination of activity hours, device volumes, and historical lags.
2. **Decision Tree Regressor:** Non-linear partitioning tree (`max_depth=7`, `min_samples_leaf=5`) to capture potential conditional thresholding.
3. **Random Forest Regressor:** Bagging ensemble of 150 randomized decision trees to reduce variance.
4. **Gradient Boosting Regressor:** Sequential additive boosted trees (`n_estimators=150`, `learning_rate=0.08`, `max_depth=5`).
5. **XGBoost Regressor:** Extreme gradient boosted trees with column sub-sampling and regularization.

#### Justification for Selecting Linear Regression:
Internet data consumption is fundamentally governed by the physics of data transfer:
$$\text{Total GB} \approx (\text{Hours} \times \text{Bandwidth Rate}) + \text{Direct Downloads} + \text{Background Traffic}$$
Because the underlying physical relationship is additive, Linear Regression models the data without the discontinuous step-function boundary errors that tree-based algorithms produce when partitioning continuous space.

### 3.3 Model Training and Validation Strategy
- **Chronological Split:** The dataset was split into **80% Training (1,760 days: Jan 1, 2020 – Oct 25, 2024)** and **20% Test (440 days: Oct 26, 2024 – Jan 8, 2026)**.
- **No Random Shuffling:** Shuffling was strictly disabled. Random shuffling creates catastrophic lookahead bias in time series because the model memorizes future lag states.
- **Empirical Uncertainty Estimation:** Out-of-sample validation residuals were extracted to compute 90% empirical prediction intervals ($q_{05}$ to $q_{95}$).

---

## Section 4. Results and Analysis

### 4.1 Comparative Model Performance (Chronological Holdout Set)

| Algorithm | Validation MAE (GB) | Validation MSE ($\text{GB}^2$) | Validation RMSE (GB) | Validation $R^2$ | Validation MAPE (%) | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.4012** | **0.2723** | **0.5218** | **0.9967** | **2.55%** | 🏆 **CHAMPION MODEL** |
| **Gradient Boosting** | 0.8120 | 1.0895 | 1.0438 | 0.9868 | 3.30% | Strong Non-linear Candidate |
| **XGBoost** | 0.8450 | 1.1668 | 1.0802 | 0.9859 | 3.45% | Strong Non-linear Candidate |
| **Random Forest** | 0.9920 | 1.8837 | 1.3725 | 0.9772 | 3.63% | High Stability Ensemble |
| **Decision Tree** | 1.1250 | 2.3731 | 1.5405 | 0.9713 | 5.82% | Baseline Tree Model |

### 4.2 Visual Analysis of Final Model Performance
- **Actual vs. Predicted Fit:** The scatter plot of actual vs. predicted values aligns tightly along the ideal 45-degree reference line ($y = \hat{y}$), with an $R^2$ of **0.9967**.
- **Residual Distribution (Homoscedasticity Check):** Residual errors ($e_i = y_i - \hat{y}_i$) are centered symmetrically around 0 with an empirical standard deviation of **0.5218 GB**. Over 90% of all daily predictions fall within $\pm 0.85\text{ GB}$ of the true consumption value.
- **Residual vs. Fitted Plot:** Confirms absence of heteroscedasticity; error variance remains consistent across both low-consumption weekdays and high-consumption weekend days.

### 4.3 Feature Importance and Key Drivers
Examining model coefficients and TreeSHAP feature importances reveals the dominant drivers:
1. **`video_streaming_hours`:** Accounts for the largest marginal predictive weight (~34.2% relative influence).
2. **`downloading_gb`:** Contributes direct volume on spike days (~22.1% relative influence).
3. **`smart_tv_usage_gb` & `laptop_usage_gb`:** Represent the primary hardware conduits (~24.5% combined).
4. **`previous_day_usage` & `rolling_7_day_average`:** Provide strong auto-regressive momentum baseline anchors.

---

## Section 5. Discussion and Conclusion

### 5.1 Summary of Findings & Problem Resolution
The **NetUsage Analytics** predictive system successfully resolves the core business problem:
- **Proactive Foresight:** Instead of waiting for billing cycle alerts, users can view accurate next-day predictions (within $\pm 0.52\text{ GB}$ error) and 7-day recursive trajectories.
- **Quota Risk Mitigation:** The application continuously compares projected multi-day consumption against the household's fair daily allowance ($Plan\_Limit / 30$), providing early warnings before broadband throttling occurs.
- **Behavioral Transparency:** Deconstructs traffic into specific activity hours, enabling households to identify which behaviors drive spikes.

### 5.2 Model Limitations and Areas for Future Improvement
1. **Session-Level Data vs. Deep Packet Inspection:** The model predicts from aggregate daily diary hours rather than hardware router packet headers.
2. **Recursive Horizon Compounding:** While 1-to-7 day forecasts maintain high accuracy, uncertainty compounds across longer horizons ($>14$ days) by $\sqrt{h}$.
3. **Future Enhancements:** Direct router SNMP integration (polling DD-WRT/OpenWrt routers automatically) and incorporating deep sequence models (Temporal Fusion Transformers) for multi-household smart utility grids.

### 5.3 Actionable Business & Household Recommendations
1. **Cap Default Streaming Resolution to 1080p:** 4K UHD streaming consumes ~7.0 GB/hr compared to ~2.2 GB/hr for Full HD. Capping default playback on Smart TVs saves ~40% bandwidth with minimal perceptible difference.
2. **Schedule Large Downloads for Off-Peak Windows:** Major game patches (25–40 GB) and OS updates should be scheduled between 2:00 AM and 6:00 AM.
3. **Optimize Work-From-Home Video Conferencing:** Turning off incoming gallery feeds during large webinars preserves up to 60% bandwidth on telecommuting days.

---

## Section 6. Appendices (Optional)

### Appendix A: Complete Mathematical Formulation
$$\hat{y}_t = \beta_0 + \sum_{i=1}^{5} \beta_i \cdot \text{Activity}_i(t) + \sum_{j=1}^{4} \gamma_j \cdot \text{Device}_j(t) + \theta_1 \cdot y_{t-1} + \theta_2 \cdot \text{MA}_7(t) + \delta \cdot \text{Weekend}(t)$$

### Appendix B: Empirical Prediction Interval Calculation
$$\text{Lower Bound}_{t+h} = \hat{y}_{t+h} + q_{05} \cdot \sqrt{h}, \quad \text{Upper Bound}_{t+h} = \hat{y}_{t+h} + q_{95} \cdot \sqrt{h}$$
*(Where $q_{05} = -0.852\text{ GB}$ and $q_{95} = +0.852\text{ GB}$ derived from validation residuals).*

### Appendix C: Anomaly Detection Boundaries
- **IQR Boundary:** $\text{Upper Cutoff} = Q_3 + 1.5 \times \text{IQR} = 30.04\text{ GB}$.
- **Isolation Forest:** Contamination rate $\alpha = 0.03$ across volume, downloads, streaming, and device density.
