#!/bin/bash

echo "============================================"
echo "Groww Algo Trading Bot - Installation Script"
echo "============================================"
echo ""

# Check Python version
python3 --version
echo ""

echo "Installing required libraries..."
echo ""

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install main requirements
pip3 install -r requirements.txt

# Install TA-Lib dependencies (Linux)
echo ""
echo "Installing TA-Lib dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential wget
    # Download and install TA-Lib
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
fi

pip3 install TA-Lib

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "Next Steps:"
echo "1. Configure your API credentials in config/credentials.py"
echo "2. Review paper_trading_config.py settings"
echo "3. Run verify_setup.py to validate installation"
echo ""
