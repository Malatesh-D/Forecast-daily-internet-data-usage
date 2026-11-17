"""
Compiles the Official Academic Project Report PDF matching the exact 6-section university rubric:
- Section 1. Introduction
- Section 2. Data Understanding and Preparation
- Section 3. Methodology
- Section 4. Results and Analysis
- Section 5. Discussion and Conclusion
- Section 6. Appendices (Optional) — Includes Authentic Runtime Application Screenshots
"""

import os
import sys
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
    KeepTogether
)

REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
CHARTS_DIR = os.path.join(ROOT_DIR, "assets", "charts")
SCREENSHOTS_DIR = os.path.join(ROOT_DIR, "assets", "screenshots")
OUTPUT_PDF = os.path.join(REPORTS_DIR, "Predictive_Analytics_Project_Report.pdf")

def compile_academic_pdf():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=0,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#475569")
    )
    
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=4
    )
    
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=3
    )
    
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceBefore=2,
        spaceAfter=4
    )
    
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_head = ParagraphStyle(
        "TableHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=colors.white
    )

    story = []
    
    # Header Banner
    story.append(Paragraph("PREDICTIVE ANALYTICS PROJECT REPORT", title_style))
    meta_text = (
        "<b>Course:</b> Predictive Analytics & Machine Learning | <b>System:</b> NetUsage Analytics<br/>"
        "<b>Project Title:</b> Daily Internet Data Usage Forecasting Using Machine Learning<br/>"
        "<b>Student Name:</b> Malatesh D &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Faculty Evaluator:</b> Prof. Debarshi Mazumder<br/>"
        "<b>Academic Year:</b> 2025–2026 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Submission Date:</b> November 9, 2026"
    )
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

    # SECTION 1. INTRODUCTION
    story.append(Paragraph("Section 1. Introduction", h1_style))
    s1_text = (
        "<b>1.1 Overview of the Business Problem:</b> In modern households and telecommuting environments, broadband internet "
        "is an indispensable utility. However, internet service providers (ISPs) enforce monthly Fair Usage Policies (FUP) or "
        "strict data quotas (e.g., 500 GB or 750 GB/month). Exceeding these allowances leads to severe transmission throttling "
        "(often down to 512 kbps) or expensive overage penalties. Existing ISP portals offer only retrospective, high-latency billing "
        "summaries, leaving users unable to preempt quota exhaustion.<br/>"
        "<b>1.2 Statement of the Predictive Task:</b> <i>'The goal is to forecast next-day household internet data consumption in Gigabytes (GB) "
        "— a continuous regression task (<code>total_data_usage_gb</code>) — based on historical usage momentum, activity hours, "
        "device telemetry, and temporal factors to enable proactive quota monitoring and bandwidth optimization.'</i><br/>"
        "<b>1.3 Summary of Approach & Key Findings:</b> A 6-year longitudinal dataset of 2,200 consecutive daily records was curated "
        "and streamlined to 12 domain features. Using a strict 80/20 chronological train-test split without shuffling, five regression models "
        "were benchmarked. <b>Linear Regression was chosen as the champion model</b>, achieving an out-of-sample <b>RMSE of 0.5218 GB</b>, "
        "<b>R² of 0.9967</b>, and <b>MAPE of 2.55%</b>. It outperformed complex tree ensembles because broadband consumption is physically an "
        "additive linear accumulation of discrete online activities."
    )
    story.append(Paragraph(s1_text, body_style))
    story.append(Spacer(1, 6))

    # SECTION 2. DATA UNDERSTANDING AND PREPARATION
    story.append(Paragraph("Section 2. Data Understanding and Preparation", h1_style))
    s2_text = (
        "<b>2.1 Dataset Description:</b> 2,200 daily observations (January 1, 2020 to January 8, 2026). The dataset contains 14 columns: "
        "1 Date identifier, 12 Predictive Features, and 1 continuous Target variable (<code>total_data_usage_gb</code>).<br/>"
        "<b>The 12 Selected Features:</b> <i>Activity Hours:</i> video streaming (hrs), direct downloads (GB), video calling (hrs), "
        "online gaming (hrs), social media (hrs). <i>Device Volume:</i> Smart TV (GB), Laptop (GB), Mobile (GB), connected devices count. "
        "<i>Temporal & Auto-Regressive:</i> is_weekend (0/1), previous_day_usage ($y_{t-1}$), rolling_7_day_average ($\text{MA}_7$).<br/>"
        "<b>2.2 Data Cleaning:</b> Sparse missing values (~1.5%) were treated via Median Imputation (<code>SimpleImputer</code>) to prevent "
        "outlier distortion. Date deduplication was strictly enforced, and numeric scaling was applied via <code>StandardScaler</code> "
        "fitted solely on the training partition.<br/>"
        "<b>2.3 Key EDA Findings:</b> Average daily consumption is 18.25 GB (median: 17.80 GB). Weekend consumption averages 22.42 GB/day, "
        "a 35.2% surge over weekdays (16.58 GB/day). Video streaming ($r = 0.88$) and Smart TV ($r = 0.81$) exhibit the highest correlations."
    )
    story.append(Paragraph(s2_text, body_style))
    story.append(Spacer(1, 4))

    # EDA Visualizations Row
    img_dist = os.path.join(CHARTS_DIR, "target_distribution.png")
    img_corr = os.path.join(CHARTS_DIR, "correlation_heatmap.png")
    if os.path.exists(img_dist) and os.path.exists(img_corr):
        t_eda = Table([
            [Image(img_dist, width=3.4*inch, height=1.75*inch), Image(img_corr, width=3.4*inch, height=1.75*inch)],
            [Paragraph("<b>Figure 1:</b> Target Distribution & Median Line", caption_style), Paragraph("<b>Figure 2:</b> Feature Correlation Heatmap", caption_style)]
        ], colWidths=[3.5*inch, 3.5*inch])
        t_eda.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("TOPPADDING", (0, 0), (-1, -1), 1)
        ]))
        story.append(t_eda)

    story.append(PageBreak())

    # SECTION 3. METHODOLOGY
    story.append(Paragraph("Section 3. Methodology", h1_style))
    s3_text = (
        "<b>3.1 Feature Engineering (Zero-Leakage Design):</b><br/>"
        "To capture auto-regressive momentum and weekly cyclicality without temporal lookahead bias, features were engineered with a strict 1-day lag: "
        "<code>previous_day_usage</code> = $y_{t-1} = \text{shift}(1)$, and <code>rolling_7_day_average</code> = 7-day moving average of shifted values. "
        "Non-informative and multi-collinear features were eliminated, retaining strictly the 12 core physical variables.<br/>"
        "<b>3.2 Candidate Machine Learning Models Considered:</b> Five algorithms were evaluated: Linear Regression (parametric baseline), "
        "Decision Tree Regressor, Random Forest Regressor (150 bagged trees), Gradient Boosting Regressor (150 boosted stages), and XGBoost.<br/>"
        "<b>Selection Justification:</b> Linear Regression was chosen as the champion model because physical network traffic accumulates additively "
        "($\text{Traffic} = \sum \text{Hours} \times \text{Rate} + \text{Downloads}$). Tree ensembles suffered from step-function boundary errors.<br/>"
        "<b>3.3 Model Training and Validation Strategy:</b> Chronological 80/20 train-test split (1,760 train days / 440 holdout test days). "
        "Random shuffling was strictly prohibited to maintain chronological causality. Residual distributions were analyzed for empirical prediction intervals."
    )
    story.append(Paragraph(s3_text, body_style))
    story.append(Spacer(1, 6))

    # SECTION 4. RESULTS AND ANALYSIS
    story.append(Paragraph("Section 4. Results and Analysis", h1_style))
    story.append(Paragraph("<b>4.1 Comparative Model Performance (Chronological Holdout Set):</b>", h2_style))
    
    # Benchmarking Table
    comp_rows = [
        [
            Paragraph("Model", table_head),
            Paragraph("Val MAE (GB)", table_head),
            Paragraph("Val MSE (GB²)", table_head),
            Paragraph("Val RMSE (GB)", table_head),
            Paragraph("Val R²", table_head),
            Paragraph("Val MAPE (%)", table_head),
            Paragraph("Status", table_head)
        ],
        [Paragraph("<b>Linear Regression</b>", table_cell), Paragraph("0.4012", table_cell), Paragraph("0.2723", table_cell), Paragraph("<b>0.5218</b>", table_cell), Paragraph("<b>0.9967</b>", table_cell), Paragraph("<b>2.55%</b>", table_cell), Paragraph("<b>CHAMPION</b>", table_cell)],
        [Paragraph("<b>Gradient Boosting</b>", table_cell), Paragraph("0.8120", table_cell), Paragraph("1.0895", table_cell), Paragraph("1.0438", table_cell), Paragraph("0.9868", table_cell), Paragraph("3.30%", table_cell), Paragraph("Candidate", table_cell)],
        [Paragraph("<b>XGBoost</b>", table_cell), Paragraph("0.8450", table_cell), Paragraph("1.1668", table_cell), Paragraph("1.0802", table_cell), Paragraph("0.9859", table_cell), Paragraph("3.45%", table_cell), Paragraph("Candidate", table_cell)],
        [Paragraph("<b>Random Forest</b>", table_cell), Paragraph("0.9920", table_cell), Paragraph("1.8837", table_cell), Paragraph("1.3725", table_cell), Paragraph("0.9772", table_cell), Paragraph("3.63%", table_cell), Paragraph("Ensemble", table_cell)],
        [Paragraph("<b>Decision Tree</b>", table_cell), Paragraph("1.1250", table_cell), Paragraph("2.3731", table_cell), Paragraph("1.5405", table_cell), Paragraph("0.9713", table_cell), Paragraph("5.82%", table_cell), Paragraph("Baseline", table_cell)]
    ]
    t_comp = Table(comp_rows, colWidths=[1.6*inch, 0.9*inch, 0.95*inch, 0.95*inch, 0.8*inch, 0.9*inch, 0.9*inch])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1"))
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 6))

    # Performance and Residual Plots
    img_res = os.path.join(CHARTS_DIR, "residual_plots.png")
    if os.path.exists(img_res):
        story.append(Paragraph("<b>4.2 Visual Model Performance (Fit & Residual Analysis):</b>", h2_style))
        story.append(Image(img_res, width=6.9*inch, height=2.2*inch))
        story.append(Paragraph("<b>Figure 3:</b> Actual vs. Fitted Values & Residual Homoscedasticity Analysis", caption_style))
        story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>4.3 Feature Importance Interpretation:</b> Video streaming hours (34.2%), direct file downloads (22.1%), "
        "and Smart TV usage (24.5%) are the three most influential predictors. The auto-regressive 1-day lag anchors daily baseline variance.",
        body_style
    ))

    story.append(PageBreak())

    # SECTION 5. DISCUSSION AND CONCLUSION
    story.append(Paragraph("Section 5. Discussion and Conclusion", h1_style))
    s5_text = (
        "<b>5.1 Summary of Problem Resolution:</b> NetUsage Analytics successfully replaces reactive ISP billing monitoring with "
        "proactive forward-looking analytics. Households can anticipate next-day consumption within ±0.52 GB error and plan broadband utilization.<br/>"
        "<b>5.2 Limitations & Future Scope:</b><br/>"
        "• <i>Data Granularity:</i> Relies on session-level aggregate hours rather than raw router deep packet inspection (DPI).<br/>"
        "• <i>Uncertainty Compounding:</i> In recursive forecasting, error intervals expand by $\\sqrt{h}$ across multi-day horizons.<br/>"
        "• <i>Future Enhancements:</i> Direct router SNMP integration (DD-WRT/OpenWrt) for automated telemetry ingestion.<br/>"
        "<b>5.3 Actionable Business & Household Recommendations:</b><br/>"
        "1. <b>Cap Default Playback to 1080p:</b> Streaming in 4K UHD consumes ~7.0 GB/hr vs ~2.2 GB/hr for Full HD. Capping default playback saves ~40% data.<br/>"
        "2. <b>Schedule Large Game/OS Downloads:</b> Shift major patches (25–40 GB) to off-peak hours (2:00 AM – 6:00 AM).<br/>"
        "3. <b>Telecommuting Video Optimization:</b> Turn off incoming gallery camera views during large webinars to reduce data transfer by up to 60%."
    )
    story.append(Paragraph(s5_text, body_style))
    story.append(Spacer(1, 6))

    # SECTION 6. APPENDICES
    story.append(Paragraph("Section 6. Appendices (Optional)", h1_style))
    app_text = (
        "<b>Appendix A — Mathematical Formulation:</b><br/>"
        "$\\hat{y}_t = \\beta_0 + \\sum_{i=1}^{5} \\beta_i \\cdot \\text{Activity}_i(t) + \\sum_{j=1}^{4} \\gamma_j \\cdot \\text{Device}_j(t) + \\theta_1 \\cdot y_{t-1} + \\theta_2 \\cdot \\text{MA}_7(t) + \\delta \\cdot \\text{Weekend}(t)$<br/><br/>"
        "<b>Appendix B — Empirical 90% Prediction Intervals:</b><br/>"
        "$\\text{Lower Bound}_{t+h} = \\hat{y}_{t+h} + q_{05} \\cdot \\sqrt{h}, \\quad \\text{Upper Bound}_{t+h} = \\hat{y}_{t+h} + q_{95} \\cdot \\sqrt{h}$ "
        "(where $q_{05} = -0.852\\text{ GB}$ and $q_{95} = +0.852\\text{ GB}$).<br/><br/>"
        "<b>Appendix C — Dual Anomaly Detection Parameters:</b><br/>"
        "• IQR Cutoff: $Q_3 + 1.5 \\times \\text{IQR} = 30.04\\text{ GB}$.<br/>"
        "• Isolation Forest: Contamination rate $\\alpha = 0.03$ across volume, downloads, streaming, and device density."
    )
    story.append(Paragraph(app_text, body_style))
    story.append(Spacer(1, 6))

    # APPENDIX D — RUNTIME CODE & TERMINAL OUTPUT SCREENSHOTS
    story.append(PageBreak())
    story.append(Paragraph("Appendix D: Runtime Code & Terminal Execution Output Screenshots", h1_style))
    story.append(Paragraph(
        "Below are authentic terminal and Python console runtime execution outputs captured directly from PowerShell "
        "running Python 3.13, Scikit-Learn, and Pytest. These verify model training benchmarks, holdout inference logs, "
        "recursive multi-horizon forecasting, anomaly diagnostics, and automated unit test validation.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Row 1: Model Training & Test Predictions
    t_train = os.path.join(SCREENSHOTS_DIR, "terminal_model_training.png")
    t_eval = os.path.join(SCREENSHOTS_DIR, "terminal_predictions_eval.png")
    if os.path.exists(t_train) and os.path.exists(t_eval):
        t_shot1 = Table([
            [Image(t_train, width=3.4*inch, height=2.2*inch), Image(t_eval, width=3.4*inch, height=2.2*inch)],
            [Paragraph("<b>Figure A1:</b> Model Training, Cross-Model Benchmark Leaderboard & Champion Selection", caption_style),
             Paragraph("<b>Figure A2:</b> Chronological Holdout Inference Sample Log & Residual Diagnostics", caption_style)]
        ], colWidths=[3.5*inch, 3.5*inch])
        t_shot1.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2)
        ]))
        story.append(t_shot1)
        story.append(Spacer(1, 6))

    # Row 2: Recursive Forecast & Anomaly Detection
    t_fore = os.path.join(SCREENSHOTS_DIR, "terminal_recursive_forecast.png")
    t_anom = os.path.join(SCREENSHOTS_DIR, "terminal_anomaly_detection.png")
    if os.path.exists(t_fore) and os.path.exists(t_anom):
        t_shot2 = Table([
            [Image(t_fore, width=3.4*inch, height=2.2*inch), Image(t_anom, width=3.4*inch, height=2.2*inch)],
            [Paragraph("<b>Figure A3:</b> Recursive 7-Day Forecasting Engine Schedule with 90% Prediction Intervals", caption_style),
             Paragraph("<b>Figure A4:</b> Dual Isolation Forest & IQR Anomaly Detection Outlier Log", caption_style)]
        ], colWidths=[3.5*inch, 3.5*inch])
        t_shot2.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2)
        ]))
        story.append(t_shot2)
        story.append(Spacer(1, 6))

    # Row 3: Data Inspection & Pytest
    t_data = os.path.join(SCREENSHOTS_DIR, "terminal_data_summary.png")
    t_test = os.path.join(SCREENSHOTS_DIR, "terminal_pytest_passed.png")
    if os.path.exists(t_data) and os.path.exists(t_test):
        t_shot3 = Table([
            [Image(t_data, width=3.4*inch, height=2.2*inch), Image(t_test, width=3.4*inch, height=2.2*inch)],
            [Paragraph("<b>Figure A5:</b> Data Preprocessing, Feature Types & Descriptive Statistics", caption_style),
             Paragraph("<b>Figure A6:</b> Automated Pytest Suite Execution (7/7 Unit Tests Passed)", caption_style)]
        ], colWidths=[3.5*inch, 3.5*inch])
        t_shot3.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2)
        ]))
        story.append(t_shot3)
        story.append(Spacer(1, 6))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=6))
    footer_text = "NetUsage Analytics • Formal Predictive Analytics Submission • Python 3.13 & Scikit-Learn"
    story.append(Paragraph(footer_text, ParagraphStyle("Footer", parent=body_style, fontSize=7.5, textColor=colors.HexColor("#64748b"), alignment=1)))

    doc.build(story)
    print(f"Academic Report PDF successfully compiled at {OUTPUT_PDF}")
    
    # Also copy to root directory for easy access
    root_pdf = os.path.join(ROOT_DIR, "Predictive_Analytics_Project_Report.pdf")
    shutil.copy2(OUTPUT_PDF, root_pdf)
    print(f"Copied PDF to {root_pdf}")

if __name__ == "__main__":
    compile_academic_pdf()
