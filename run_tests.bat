@echo off
title NetUsage Analytics - Automated Test Suite
echo ========================================================
echo   Running NetUsage Analytics Test Suite
echo ========================================================
echo.

cd /d "%~dp0"
python -m unittest tests/test_system.py

echo.
echo ========================================================
pause
