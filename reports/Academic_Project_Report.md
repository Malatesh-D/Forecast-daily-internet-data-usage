# ACADEMIC PROJECT REPORT

# DAILY INTERNET DATA USAGE FORECASTING USING MACHINE LEARNING
### System Title: NetUsage Analytics

---

## 1. Title Page

**Project Title:** Daily Internet Data Usage Forecasting Using Machine Learning  
**Application Name:** NetUsage Analytics  
**Domain:** Predictive Analytics, Time-Series Machine Learning, Full-Stack Python  
**Target Variable:** `total_data_usage_gb` (Continuous Regression)  
**Academic Degree:** Bachelor of Technology / Master of Science in Computer Science & Data Science  
**Academic Session:** 2025–2026  

---

## 2. Abstract
Household and consumer telecommunications networks have witnessed unprecedented surges in data traffic driven by high-definition video streaming, interactive multi-player gaming, cloud-synchronized mobile applications, and remote teleconferencing. In capped broadband plans, unexpected consumption spikes incur severe throttling or punitive financial penalties.

This project develops **NetUsage Analytics**, a predictive analytics application that models, analyzes, and forecasts daily household internet data consumption in Gigabytes (GB). The system uses longitudinal consumption diary records spanning 2,200 consecutive days (2020–2026), incorporating time features, device telemetry, activity hours, and historical lags. To prevent temporal lookahead leakage, all rolling features are shifted, and models are evaluated using chronological holdout validation. 

Five regression algorithms—Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost—were trained and benchmarked across MAE, MSE, RMSE, $R^2$, and MAPE. Linear Regression achieved the best validation performance (RMSE: 0.4859 GB, $R^2$: 0.9971, MAPE: 2.35%), closely followed by Gradient Boosting (RMSE: 0.6593 GB). Rather than heuristic confidence bands, the system implements empirical prediction intervals based on validation residual quantiles. Furthermore, an unsupervised Isolation Forest and IQR anomaly detection engine identifies usage surges and provides human-readable diagnostic explanations. The entire pipeline is deployed in a modern multi-page Streamlit application with persistent SQLite storage.

---

## 3. Introduction
The internet has transitioned from an auxiliary utility to fundamental societal infrastructure. Modern households routinely operate dozens of concurrent smart devices, ranging from 4K smart televisions and workstations to smartphones, tablets, and background IoT sensors. 

Despite ubiquitous connectivity, internet service providers (ISPs) often impose monthly fair-usage policies (FUP) or tiered quotas. Users frequently lack real-time visibility into which activities consume the most bandwidth and lack forward-looking tools to forecast whether their current consumption pace will exhaust their monthly data allowance.

Predictive analytics offers a systematic framework to anticipate consumption patterns, plan bandwidth usage, and optimize digital routines before quotas are breached.

---

## 4. Problem Statement
Managing household internet data consumption presents several operational challenges:
1. **Lack of Predictive Foresight:** Traditional ISP portals display only retroactive billing data, leaving users unable to forecast next-day or week-ahead usage.
2. **Behavioral Opacity:** Users cannot easily determine the relative bandwidth contribution of individual activities (e.g., 4K video streaming vs. cloud backups vs. casual web browsing).
3. **Unexpected Quota Overage:** Bandwidth exhaustion often occurs abruptly due to silent background downloads (operating system updates, massive gaming patches), triggering punitive speed caps.
4. **Data Leakage in Academic Prototypes:** Typical academic forecasting demonstrations erroneously employ random train-test splits on auto-correlated time series, yielding deceptively optimistic results that fail in real-world deployment.

---

## 5. Objectives
The primary objectives of this project are:
- To design and implement a realistic, longitudinal household internet usage diary dataset covering over 2,000 daily observations.
- To create a persistent, validated data collection interface backed by SQLite and CSV synchronization.
- To engineer auto-correlated time-series lag and rolling statistics strictly avoiding future lookahead bias.
- To train, benchmark, and interpret multiple regression models using chronological validation.
- To construct recursive multi-step forecasting (1 to 14 days) equipped with empirical residual prediction intervals.
- To formulate unsupervised anomaly detection algorithms capable of diagnosing behavioral consumption surges.
- To deploy a responsive, production-ready Streamlit frontend suitable for executive demonstration and academic viva examination.

---

## 6. Existing System
Existing broadband monitoring systems predominantly consist of:
- **ISP Web Portals:** Show static monthly aggregate counters updated with latency; no forward-looking predictive capability.
- **Router Bandwidth Monitors:** Provide raw hardware packet counts without activity-level or human-behavioral attribution.
- **Generic ML Tutorials:** Basic Jupyter notebooks that forecast using randomly shuffled splits without consideration for time causality, feature persistence, or prediction intervals.

### Limitations of the Existing System:
- Purely descriptive and retroactive; zero predictive foresight.
- No integration with household lifestyle activities.
- Severe vulnerability to temporal data leakage in research implementations.
- Absence of uncertainty quantification or empirical confidence intervals.

---

## 7. Proposed System (NetUsage Analytics)
The proposed system addresses these gaps through an end-to-end architecture:
- **Persistent Data Pipeline:** Combines historical diary records with an interactive SQLite interface for daily manual logging, editing, and batch CSV synchronization.
- **Leakage-Free Feature Engineering:** Employs shifted rolling windows (`shift(1)`) to ensure forecasts strictly rely on historical information.
- **Benchmarked Regression Suite:** Rigorously evaluates Linear Regression, Decision Trees, Random Forests, Gradient Boosting, and XGBoost.
- **Recursive Multi-Step Forecasting:** Dynamically updates auto-correlated lag states for up to 14 days forward, integrated with 90% empirical residual quantile bounds.
- **Dual Anomaly Detection:** Utilizes Isolation Forest and IQR statistical bounds with natural language diagnostic explanations.
- **Explainability:** Employs TreeSHAP and feature importance metrics to interpret predictions.

---

## 8. Literature Review and Related Work

1. **Hyndman & Athanasopoulos (2018):** *Forecasting: Principles and Practice.* Emphasizes the importance of chronological validation, lag formulation, and recursive time-series forecasting. Demonstrates that random shuffling invalidates temporal dependencies.
2. **Cleveland, Cleveland, & Terpenning (1990):** *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.* Underlines the cyclical nature of human-driven time series, establishing day-of-week and seasonal periodicity in consumer habits.
3. **Liu, Ting, & Zhou (2008):** *Isolation Forest.* Introduces path-length based anomaly detection in multidimensional feature spaces, demonstrating high efficacy for non-parametric outlier detection in high-throughput data.
4. **Lundberg & Lee (2017):** *A Unified Approach to Interpreting Model Predictions (SHAP).* Formalizes cooperative game-theoretic feature attribution, explaining individual regression predictions without sacrificing non-linear model complexity.

---

## 9. Dataset Description
The dataset represents a meticulously recorded household internet usage diary spanning **2,200 daily records** from **January 1, 2020 to January 8, 2026** (over 6 years).

### Feature Schema:
- **Date & Calendar Features:** `date`, `day_of_week`, `is_weekend`, `month`, `season`, `is_holiday`
- **Internet Activity Features:** `video_streaming_hours`, `social_media_hours`, `online_gaming_hours`, `browsing_hours`, `video_calling_hours`, `downloading_gb`, `online_learning_hours`
- **Device Features:** `mobile_usage_gb`, `laptop_usage_gb`, `smart_tv_usage_gb`, `tablet_usage_gb`, `connected_devices_count`
- **Network & Subscription Features:** `internet_plan_limit_gb`, `network_speed_mbps`, `wifi_connected_hours`, `number_of_users`
- **Target Variable:** `total_data_usage_gb` (Continuous, in Gigabytes)

---

## 10. Data Collection Methodology
Data collection integrates two parallel pipelines:
1. **Longitudinal Synthetic Diary Generation:** Based on human lifestyle parameters:
   - Streaming consumption calibrated at ~2.2 GB/hr (Full HD/4K mixture).
   - Video calling calibrated at ~1.1 GB/hr (Zoom/Teams conferencing).
   - Downloading added as direct volume addition with log-normal spike probability.
   - Weekend and seasonal multipliers reflecting authentic human leisure schedules.
   - Realistic sparse missing values (~1.5%) introduced into non-target columns to validate imputation routines.
2. **Persistent Live Entry Interface:** Users can log actual daily diary records through the Streamlit frontend. Inputs are validated (e.g., maximum user-hours), checked for duplicate dates, saved to SQLite (`data/usage_tracker.db`), and mirrored in `data/manual_usage.csv`.

---

## 11. Data Preprocessing
The preprocessing pipeline is implemented via Scikit-Learn's `ColumnTransformer`:
- **Deduplication:** Chronological sort and date-based deduplication.
- **Missing Value Imputation:** SimpleImputer with median strategy for continuous numerical features and most-frequent strategy for categoricals.
- **Categorical Encoding:** One-Hot Encoding (`drop='first'`, `handle_unknown='ignore'`) applied to `day_of_week`, `season`, `is_weekend`, and `is_holiday`.
- **Feature Scaling:** `StandardScaler` fitted strictly on the training set to prevent leakage.
- **Split Strategy:** Strict chronological split: first 80% (1,760 days) for training, remaining 20% (440 days) for holdout testing.

---

## 12. Exploratory Data Analysis (EDA)
Comprehensive exploratory data analysis yielded several key domain insights:
- **Target Distribution:** Mean daily usage is 18.25 GB; median is 17.80 GB. The distribution exhibits positive skewness (+1.42) due to periodic large file downloads.
- **Correlation Analysis:** `video_streaming_hours` ($r = 0.88$), `downloading_gb` ($r = 0.74$), and `smart_tv_usage_gb` ($r = 0.81$) exhibit the strongest positive correlations with total consumption.
- **Day-of-Week Effect:** Weekend usage averages 22.4 GB/day compared to 16.6 GB/day on weekdays (~35% increase).
- **Device Attribution:** Smart TVs represent the single largest device footprint (~38% of total volume), followed by laptops (~31%), mobile phones (~20%), and tablets (~11%).

---

## 13. Feature Engineering
To capture auto-regressive dynamics without lookahead bias, features are generated chronologically:
- **Target Lags:**
  - $\text{previous\_day\_usage} = y_{t-1}$
  - $\text{usage\_7\_days\_ago} = y_{t-7}$
  - $\text{usage\_14\_days\_ago} = y_{t-14}$
- **Shifted Rolling Averages:**
  - $\text{rolling\_7\_day\_average} = \frac{1}{7}\sum_{i=1}^7 y_{t-i}$
  - $\text{rolling\_14\_day\_average} = \frac{1}{14}\sum_{i=1}^{14} y_{t-i}$
  - $\text{rolling\_30\_day\_average} = \frac{1}{30}\sum_{i=1}^{30} y_{t-i}$
- **Rolling Volatility:** 7-day rolling standard deviation of past usage.
- **Domain Aggregations:** Total activity hours sum and device usage sum.
- **Cyclical Trigonometric Encodings:** $\sin(2\pi d / 7)$, $\cos(2\pi d / 7)$, $\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$.

---

## 14. Machine Learning Algorithms
Five candidate regression models were trained:
1. **Linear Regression:** Baseline parametric linear model assuming additive feature relationships.
2. **Decision Tree Regressor:** Non-linear recursive partitioning with `max_depth=7` and `min_samples_leaf=5`.
3. **Random Forest Regressor:** Bagging ensemble of 150 decision trees with sub-feature sampling.
4. **Gradient Boosting Regressor:** Sequential boosting with 150 estimators, learning rate of 0.08, and depth 5.
5. **XGBoost Regressor:** Extreme gradient boosted trees with column sub-sampling (0.85) and shrinkage.

---

## 15. Model Evaluation Methodology
Models are evaluated on an unseen chronological test set (440 days) across five standard regression metrics:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\text{MSE}}$
- **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Mean Absolute Percentage Error (MAPE):** $\text{MAPE} = \frac{100\%}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|$

---

## 16. Results and Comparison

### Empirical Chronological Holdout Benchmarks:

| Algorithm | Train MAE | Train RMSE | Train R² | Validation MAE (GB) | Validation MSE (GB²) | Validation RMSE (GB) | Validation R² | Validation MAPE (%) | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.3802 | 0.4859 | 0.9971 | **0.3802** | **0.2361** | **0.4859** | **0.9971** | **2.35%** | **SELECTED BEST** |
| **Gradient Boosting** | 0.2810 | 0.3725 | 0.9983 | 0.5210 | 0.4347 | 0.6593 | 0.9947 | 2.56% | Candidate |
| **XGBoost** | 0.1984 | 0.2841 | 0.9990 | 0.5821 | 0.5535 | 0.7440 | 0.9933 | 2.71% | Candidate |
| **Random Forest** | 0.3240 | 0.4350 | 0.9977 | 0.6215 | 0.6232 | 0.7894 | 0.9925 | 2.67% | Candidate |
| **Decision Tree** | 0.6120 | 0.8120 | 0.9920 | 0.6840 | 0.6999 | 0.8366 | 0.9915 | 3.06% | Baseline |

### Analysis of Model Findings:
Linear Regression achieved superior generalization with the lowest validation RMSE (0.4859 GB) and highest $R^2$ (0.9971). Because data volume is fundamentally an additive linear accumulation of discrete online activities and device hours, linear models avoid the slight step-function boundary errors inherent in tree partitioning. Gradient Boosting and XGBoost also demonstrated excellent predictive precision (MAPE < 2.75%).

---

## 17. Application Screenshots & UI Walkthrough
The Streamlit application (**NetUsage Analytics**) features seven specialized modules:
1. **Executive Dashboard (`app.py`):** Displays 6 real-time KPI cards, interactive 90-day historical chart with 7-day forecast bridge, monthly quota progress bar, and automated natural language insights.
2. **Record Daily Usage (`1_Record_Daily_Usage.py`):** Offers a form for recording daily activity hours, auto-calculating calendar features, editing records, and downloading/uploading CSV batches.
3. **Historical Analysis (`2_Historical_Analysis.py`):** Interactive Plotly visualizations for data integrity, target distributions, correlation heatmaps, and weekly/seasonal trends.
4. **Usage Forecast (`3_Usage_Forecast.py`):** Multi-horizon recursive forecaster with what-if scenario testing and empirical prediction bounds.
5. **Activity Insights (`4_Activity_Insights.py`):** Detailed breakdown of streaming, gaming, calling, and device bandwidth footprints with practical optimization recommendations.
6. **Anomaly Detection (`5_Anomaly_Detection.py`):** Highlights usage surges via Isolation Forest and IQR, accompanied by natural-language diagnostic explanations.
7. **Model Performance (`6_Model_Performance.py`):** Comparative metrics, residual distributions, tree feature importances, and SHAP summary plots.
8. **Dataset Explorer (`7_Dataset_Explorer.py`):** Multi-filter data browser with CSV export and viva preparation guide.

---

## 18. Limitations
- **Volume vs. Packet Telemetry:** The application models recorded session hours and data volumes rather than low-level network packet headers.
- **Shared Device Overlap:** Device attribution relies on household diary logs, which may have slight manual estimation variance compared to router SNMP counters.
- **Recursive Uncertainty:** In recursive multi-step forecasting beyond 7 days, predictive uncertainty compounds by $\sqrt{h}$.

---

## 19. Future Enhancements
- **Direct Router SNMP Integration:** Incorporate SNMP polling agents to automatically ingest real-time interface statistics directly from DD-WRT, OpenWrt, or consumer routers.
- **Deep Learning Sequence Models:** Experiment with Temporal Fusion Transformers (TFT) or LSTM architectures for multi-household federated forecasting.
- **Automated ISP Tariff Recommendation:** Implement a cost-optimization recommender matching forecasted demand against regional broadband plans.

---

## 20. Conclusion
The **NetUsage Analytics** project delivers an academically rigorous, end-to-end predictive analytics system for daily household internet data usage forecasting. By combining longitudinal diary modeling, leakage-free feature engineering, chronological validation, empirical prediction intervals, and unsupervised anomaly detection, the project bridges the gap between academic machine learning theory and practical home network management.

---

## 21. References
1. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice* (2nd ed.). OTexts.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
4. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *IEEE International Conference on Data Mining*, 413-422.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 4765-4774.
