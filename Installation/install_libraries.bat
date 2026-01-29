@echo off
echo ============================================
echo Groww Algo Trading Bot - Installation Script
echo ============================================
echo.

REM Check Python version
python --version
echo.

echo Installing required libraries...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install main requirements
pip install -r requirements.txt

REM Install TA-Lib (may require pre-built wheel on Windows)
echo.
echo NOTE: If TA-Lib installation fails, you may need to install it manually:
echo 1. Download the appropriate wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo 2. Run: pip install TA_Lib-0.4.25-cp311-cp311-win_amd64.whl
echo.

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo Next Steps:
echo 1. Configure your API credentials in config/credentials.py
echo 2. Review paper_trading_config.py settings
echo 3. Run verify_setup.py to validate installation
echo.
pause
