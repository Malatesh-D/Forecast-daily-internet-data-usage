@echo off
title NetUsage Analytics - Streamlit Dashboard
echo ========================================================
echo   Launching NetUsage Analytics Executive Dashboard
echo   Theme: Clean Pure White | Features: Streamlined (12)
echo ========================================================
echo.

cd /d "%~dp0"
streamlit run app.py

pause
