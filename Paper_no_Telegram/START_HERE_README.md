# Groww Algo Trading Bot

A comprehensive algorithmic trading bot for **Groww Broker** platform, adapted from the Dhan_Algo_New project structure.

## Quick Links

- **Groww API Documentation**: https://groww.in/trade-api/docs/python-sdk
- **PyPI Package**: https://pypi.org/project/growwapi/
- **Package Version**: 1.5.0+

## Installation

### 1. Install Python Dependencies

```bash
cd Installation
pip install -r requirements.txt
```

Or run the installation script:
```bash
chmod +x install_libraries.sh
./install_libraries.sh
```

### 2. Get Groww API Credentials

1. Go to [Groww Cloud API Keys Page](https://groww.in/trade-api)
2. Log in to your Groww account
3. Click 'Generate API key'
4. Copy your API Key and Secret

### 3. Configure Credentials

Edit `single_trade_focus_bot.py` and update:
```python
API_KEY = "YOUR_GROWW_API_KEY"
API_SECRET = "YOUR_GROWW_API_SECRET"
```

## Project Structure

```
Groww_Algo/
├── Installation/
│   ├── requirements.txt         # Python dependencies
│   └── install_libraries.sh     # Installation script
│
└── Paper_no_Telegram/
    ├── single_trade_focus_bot.py    # Main trading bot
    ├── Groww_Tradehull.py           # Groww API wrapper
    │
    ├── ENTRY/EXIT LOGIC
    ├── ce_entry_logic.py            # Call option entry
    ├── pe_entry_logic.py            # Put option entry
    ├── exit_logic.py                # Exit management
    │
    ├── TECHNICAL INDICATORS
    ├── VWAP.py                      # Volume Weighted Avg Price
    ├── ATRTrailingStop.py           # ATR trailing stops
    ├── adx_indicator.py             # ADX indicator
    ├── Fractal_Chaos_Bands.py       # Fractal bands
    │
    ├── PAPER TRADING
    ├── paper_trading_config.py      # Configuration
    ├── paper_trading_simulator.py   # Simulator
    ├── paper_trading_wrapper.py     # Auto-switch wrapper
    │
    ├── UTILITIES
    ├── websocket_manager.py         # Live data streaming
    ├── rate_limiter.py              # API rate limiting
    └── SectorPerformanceAnalyzer.py # Watchlist generation
```

## Usage

### Paper Trading (Recommended First)

1. Ensure paper trading is enabled in `paper_trading_config.py`:
   ```python
   PAPER_TRADING_ENABLED = True
   ```

2. Run the bot:
   ```bash
   python single_trade_focus_bot.py
   ```

### Live Trading

1. Disable paper trading:
   ```python
   PAPER_TRADING_ENABLED = False
   ```

2. Run the bot:
   ```bash
   python single_trade_focus_bot.py
   ```

## Groww API Quick Reference

```python
from growwapi import GrowwAPI, GrowwFeed

# Initialize
client = GrowwAPI("YOUR_API_KEY")

# Get orders
orders = client.get_order_list(timeout=5)

# Live data
feed = GrowwFeed("YOUR_API_KEY")
feed.subscribe_live_data(GrowwAPI.SEGMENT_CASH, "RELIANCE")
ltp = feed.get_stocks_ltp("RELIANCE", timeout=3)
```

## Risk Management

Default settings in `single_trade_focus_bot.py`:
- Max orders per day: 2
- Max simultaneous positions: 2
- Option stop loss: 15%
- ATR multiplier: 3
- Risk/Reward ratio: 3

## Support

For API issues, contact: growwapi@groww.in
