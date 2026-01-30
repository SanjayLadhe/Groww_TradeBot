# Groww Algo Trading Bot (RL-Enhanced)

A comprehensive algorithmic trading bot for **Groww Broker** platform with **Reinforcement Learning** optimization.

## New: RL Enhancement

The bot now includes an optional **Reinforcement Learning layer** that optimizes:
- **Entry decisions** - Whether to take a signal or skip it
- **Position sizing** - How aggressively to trade (lot multiplier)
- **Risk parameters** - Adaptive stop loss and target adjustments
- **Exit timing** - When to exit early or let profits run

Your base strategy (VWAP, ADX, ATR, Fractal) remains unchanged. The RL acts as a **meta-layer** that learns to optimize decisions on top of your strategy.

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
│   ├── requirements.txt         # Python dependencies (incl. PyTorch)
│   └── install_libraries.sh     # Installation script
│
└── Paper_no_Telegram/
    ├── single_trade_focus_bot.py    # Original trading bot (no RL)
    ├── rl_enhanced_bot.py           # 🆕 RL-enhanced trading bot
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
    ├── REINFORCEMENT LEARNING (NEW)
    ├── rl_config.py                 # 🆕 RL configuration
    ├── rl_state_builder.py          # 🆕 State vector builder
    ├── rl_agent.py                  # 🆕 PPO agent & neural network
    ├── rl_reward.py                 # 🆕 Reward functions
    ├── rl_trainer.py                # 🆕 Training script
    │
    ├── PAPER TRADING
    ├── paper_trading_config.py      # Configuration
    ├── paper_trading_simulator.py   # Simulator
    ├── paper_trading_wrapper.py     # Auto-switch wrapper
    │
    ├── UTILITIES
    ├── websocket_manager.py         # Live data streaming
    ├── rate_limiter.py              # API rate limiting
    ├── trade_logger.py              # Excel export & logging
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

## Reinforcement Learning Mode

### Quick Start (RL Mode)

1. **Train the RL agent** (optional, can also learn online):
```bash
python rl_trainer.py --mode train --episodes 1000
```

2. **Run RL-enhanced bot**:
```bash
python rl_enhanced_bot.py
```

### RL Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL-ENHANCED TRADING SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  MARKET DATA    │───▶│  YOUR STRATEGY  │───▶│   SIGNAL    │ │
│  │  (OHLCV)        │    │  (VWAP,ADX,ATR) │    │   (CE/PE)   │ │
│  └─────────────────┘    └─────────────────┘    └──────┬──────┘ │
│                                                        │        │
│                                                        ▼        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     RL META-LAYER (PPO)                     ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  STATE:                                                     ││
│  │  • Technical indicators (VWAP, ADX, ATR, Fractal)          ││
│  │  • Price action features                                    ││
│  │  • Position state (P&L, duration, distance to SL)          ││
│  │  • Session stats (daily P&L, win rate)                     ││
│  │  • Time features (time of day, time to close)              ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  ACTIONS:                                                   ││
│  │  • Take signal or Skip (discrete)                          ││
│  │  • Position size multiplier: 0.5x - 2.0x                   ││
│  │  • Stop loss adjustment: 0.8x - 1.2x                       ││
│  │  • Target adjustment: 0.8x - 1.5x                          ││
│  │  • Trailing stop factor: 0.5x - 1.5x                       ││
│  └──────────────────────────────┬──────────────────────────────┘│
│                                  │                               │
│                                  ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    OPTIMIZED TRADE EXECUTION                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### RL Configuration (`rl_config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RL_ENABLED` | True | Enable/disable RL optimization |
| `RL_ALGORITHM` | PPO | Algorithm (PPO recommended) |
| `LEARNING_RATE` | 3e-4 | Neural network learning rate |
| `MIN_CONFIDENCE_TO_ACT` | 0.6 | Minimum confidence to take signal |
| `HIDDEN_LAYERS` | [256, 128, 64] | Neural network architecture |

### RL Files

| File | Purpose |
|------|---------|
| `rl_config.py` | RL configuration and hyperparameters |
| `rl_state_builder.py` | Builds state vector from market data |
| `rl_agent.py` | PPO agent and neural network |
| `rl_reward.py` | Reward calculation functions |
| `rl_enhanced_bot.py` | Main RL-enhanced trading bot |
| `rl_trainer.py` | Offline training script |

### Training Modes

**1. Online Learning (Default)**
The agent learns while paper trading in real-time.

**2. Offline Training**
Train on synthetic data before live trading:
```bash
python rl_trainer.py --mode train --episodes 1000
```

**3. Evaluation**
Test a trained model:
```bash
python rl_trainer.py --mode evaluate --model models/best_model.pt
```

### Safety Features

The RL cannot override these hard limits:
- Max position size: 10 lots
- Stop loss range: 5% - 30%
- Target range: 5% - 100%
- Max daily trades: 5
- Confidence threshold: 60%

## Support

For API issues, contact: growwapi@groww.in
