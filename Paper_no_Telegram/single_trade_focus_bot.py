"""
Groww Algo Trading Bot - Single Trade Focus Bot

This is the main trading bot for the Groww broker platform.
Adapted from Dhan_Algo_New for compatibility with Groww API.

Features:
- Multi-position trading (configurable max simultaneous positions)
- Dynamic watchlist from sector performance analysis
- Technical indicator-based entry signals (VWAP, ATR, ADX, Fractal)
- Options trading (CE/PE) with risk management
- Stop loss and trailing stop loss management
- Paper trading mode for testing
- Detailed console logging with emoji indicators

Groww API Documentation: https://groww.in/trade-api/docs/python-sdk
Package: pip install growwapi (version 1.5.0+)

Author: Algo Trading Bot
Date: 2026
"""

# =============================================================================
# IMPORTS
# =============================================================================

import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Groww API Wrapper
from Groww_Tradehull import Tradehull

# Entry/Exit Logic Modules
from ce_entry_logic import execute_ce_entry, check_ce_entry_conditions
from pe_entry_logic import execute_pe_entry, check_pe_entry_conditions
from exit_logic import check_exit_conditions, execute_exit, update_trailing_stop

# Technical Indicators
from VWAP import calculate_vwap
from ATRTrailingStop import calculate_atr, calculate_atr_trailing_stop
from adx_indicator import calculate_adx
from Fractal_Chaos_Bands import calculate_fractal_chaos_bands

# Utility Modules
from rate_limiter import RateLimiter, get_data_api_limiter, get_order_api_limiter
from websocket_manager import WebSocketManager
from SectorPerformanceAnalyzer import get_dynamic_watchlist

# Paper Trading Config
from paper_trading_config import PAPER_TRADING_ENABLED, PAPER_TRADING_BALANCE

# Trade Logger for Excel export
from trade_logger import TradeLogger, init_trade_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trade logger instance
trade_logger: TradeLogger = None

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Credentials (Replace with your actual credentials)
# Get your API key from: https://groww.in/trade-api
API_KEY = "YOUR_GROWW_API_KEY"
API_SECRET = "YOUR_GROWW_API_SECRET"  # Optional

# Trading Parameters
OPENING_BALANCE = 1005000           # Account opening balance
BASE_CAPITAL = 1000000              # Base capital for risk calculation
MARKET_MONEY_RISK_PCT = 0.01        # 1% risk on market gains
BASE_CAPITAL_RISK_PCT = 0.005       # 0.5% risk on base capital

MAX_ORDERS_TODAY = 2                # Maximum orders per day
MAX_SIMULTANEOUS_POSITIONS = 2      # Maximum simultaneous positions

RISK_PER_TRADE = None               # Calculated dynamically

ATR_MULTIPLIER = 3                  # ATR multiplier for stop loss
RISK_REWARD_RATIO = 3               # Risk to reward ratio
OPTION_SL_PERCENTAGE = 0.15         # 15% stop loss for options

# Timing Configuration
MONITOR_INTERVAL = 5                # Seconds between position checks
SCAN_INTERVAL = 30                  # Seconds between entry scans
WATCHLIST_REFRESH_INTERVAL = 900    # 15 minutes (900 seconds)
BATCH_SIZE = 5                      # Symbols per batch for scanning

# Market Hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# No new trades after this time
NO_NEW_TRADE_HOUR = 14
NO_NEW_TRADE_MINUTE = 45

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Trading instance
tsl = None

# Active positions tracking
active_positions: Dict[str, Dict] = {}

# Today's statistics
todays_orders = 0
todays_pnl = 0.0

# Watchlist
watchlist: List[str] = []
last_watchlist_refresh = None

# WebSocket manager
ws_manager = None

# Rate limiters
data_limiter = None
order_limiter = None

# =============================================================================
# BANNER AND DISPLAY FUNCTIONS
# =============================================================================

def print_startup_banner():
    """Print startup banner with mode information."""
    print("\n" + "=" * 100)
    if PAPER_TRADING_ENABLED:
        print("🎮 PAPER TRADING MODE ENABLED")
        print("=" * 100)
        print("⚠️  NO REAL ORDERS WILL BE PLACED")
        print("⚠️  ALL TRADES ARE SIMULATED")
        print(f"💰 Starting Balance: ₹{PAPER_TRADING_BALANCE:,.2f}")
        print(f"📝 Logs: paper_trading_log.txt")
    else:
        print("🚀 LIVE TRADING MODE")
        print("=" * 100)
        print("⚠️  REAL MONEY WILL BE USED")
        print(f"💰 Opening Balance: ₹{OPENING_BALANCE:,.2f}")
    print("=" * 100 + "\n")


def print_bot_started_banner():
    """Print bot started banner with configuration."""
    print("\n" + "=" * 100)
    print("🚀 GROWW SINGLE TRADE FOCUS BOT STARTED")
    print("=" * 100)
    print(f"📊 Strategy: Focus on ONE high-quality trade at a time")
    print(f"⏰ Scan Interval (No Position): {SCAN_INTERVAL} seconds")
    print(f"👁️  Monitor Interval (In Position): {MONITOR_INTERVAL} seconds")
    print(f"🔄 Watchlist Refresh Interval: {WATCHLIST_REFRESH_INTERVAL // 60} minutes")
    print(f"💰 Risk per Trade: ₹{RISK_PER_TRADE:,.2f}")
    print(f"📈 Risk Reward Ratio: 1:{RISK_REWARD_RATIO}")
    print(f"📋 Watchlist: {len(watchlist)} symbols")
    print("=" * 100 + "\n")


def print_market_status():
    """Print current market status."""
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')

    if is_market_hours():
        print(f"\n{'=' * 80}")
        print(f"📈 MARKET OPEN at {time_str}")
        print(f"{'=' * 80}")
    else:
        print(f"\n{'=' * 80}")
        print(f"🔔 MARKET CLOSED at {time_str}")
        print(f"{'=' * 80}")


def print_scan_header():
    """Print scanning header."""
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    current_positions = len(active_positions)
    max_positions = MAX_SIMULTANEOUS_POSITIONS

    print(f"\n{'=' * 80}")
    print(f"🔍 SCANNING FOR ENTRY - {time_str}")
    print(f"{'=' * 80}")
    print(f"📊 Current Positions: {current_positions}/{max_positions}")
    print(f"📋 Watchlist Size: {len(watchlist)} symbols")
    print()


def print_monitor_header(symbol: str):
    """Print position monitoring header."""
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')

    print(f"\n{'=' * 80}")
    print(f"👁️  MONITORING ACTIVE POSITION: {symbol} - {time_str}")
    print(f"{'=' * 80}")


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_bot():
    """Initialize the trading bot."""
    global tsl, ws_manager, data_limiter, order_limiter, watchlist
    global RISK_PER_TRADE, trade_logger

    print("\nGroww Algo Bot Version 1.0")
    print("-----Connecting to Groww-----")

    # Initialize trade logger for Excel export
    trade_logger = init_trade_logger(log_dir="logs", excel_file="Live Trade Data.xlsx")

    # Calculate risk parameters
    market_gains = OPENING_BALANCE - BASE_CAPITAL
    market_risk = market_gains * MARKET_MONEY_RISK_PCT if market_gains > 0 else 0
    base_risk = BASE_CAPITAL * BASE_CAPITAL_RISK_PCT
    max_risk_today = market_risk + base_risk
    RISK_PER_TRADE = max_risk_today / MAX_ORDERS_TODAY

    # Initialize Groww API
    if PAPER_TRADING_ENABLED:
        from paper_trading_wrapper import get_trading_instance
        tsl = get_trading_instance(API_KEY, API_SECRET)
        print("[PAPER] Paper Trading Simulator Initialized")
    else:
        tsl = Tradehull(API_KEY, API_SECRET)
        print("-----Logged into Groww-----")

    # Initialize rate limiters
    data_limiter = get_data_api_limiter()
    order_limiter = get_order_api_limiter()

    # Initialize WebSocket manager
    ws_manager = WebSocketManager(API_KEY)

    # Get initial watchlist
    print("Fetching watchlist from best performing sectors...")
    watchlist = get_dynamic_watchlist()
    print(f"✅ Using sector-based watchlist with {len(watchlist)} stocks\n")

    return True


def is_market_hours() -> bool:
    """Check if current time is within market hours."""
    now = datetime.now()
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)

    return market_open <= now <= market_close


def can_place_new_trade() -> bool:
    """Check if new trades are allowed."""
    now = datetime.now()
    cutoff = now.replace(hour=NO_NEW_TRADE_HOUR, minute=NO_NEW_TRADE_MINUTE, second=0)

    if now > cutoff:
        return False

    if todays_orders >= MAX_ORDERS_TODAY:
        return False

    if len(active_positions) >= MAX_SIMULTANEOUS_POSITIONS:
        return False

    return True


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def add_position(
    symbol: str,
    order_id: str,
    entry_price: float,
    quantity: int,
    option_type: str,
    stop_loss: float,
    target: float
) -> None:
    """Add a new position to tracking."""
    global todays_orders

    active_positions[symbol] = {
        "order_id": order_id,
        "entry_price": entry_price,
        "quantity": quantity,
        "option_type": option_type,
        "stop_loss": stop_loss,
        "target": target,
        "trailing_stop": stop_loss,
        "entry_time": datetime.now(),
        "highest_price": entry_price,
        "lowest_price": entry_price
    }

    todays_orders += 1

    # Log to trade logger for Excel export
    if trade_logger:
        trade_logger.log_order(
            order_id=order_id,
            symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            order_type="MARKET",
            price=entry_price,
            status="COMPLETE",
            option_type=option_type
        )
        trade_logger.log_trade(
            trade_id=f"TRD_{order_id}",
            order_id=order_id,
            symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            executed_price=entry_price
        )
        trade_logger.log_position_open(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            option_type=option_type
        )


def remove_position(symbol: str, exit_price: float, reason: str) -> float:
    """Remove a position and calculate P&L."""
    global todays_pnl

    if symbol not in active_positions:
        print(f"⚠️ Position not found: {symbol}")
        return 0.0

    position = active_positions[symbol]
    entry_price = position["entry_price"]
    quantity = position["quantity"]

    pnl = (exit_price - entry_price) * quantity
    todays_pnl += pnl

    # Log to trade logger for Excel export
    if trade_logger:
        trade_logger.log_order(
            order_id=f"EXIT_{position['order_id']}",
            symbol=symbol,
            transaction_type="SELL",
            quantity=quantity,
            order_type="MARKET",
            price=exit_price,
            status="COMPLETE",
            exit_reason=reason
        )
        trade_logger.log_trade(
            trade_id=f"TRD_EXIT_{position['order_id']}",
            order_id=f"EXIT_{position['order_id']}",
            symbol=symbol,
            transaction_type="SELL",
            quantity=quantity,
            executed_price=exit_price
        )
        trade_logger.log_position_close(
            symbol=symbol,
            exit_price=exit_price,
            exit_reason=reason
        )

    # Unsubscribe from WebSocket
    if ws_manager:
        ws_manager.unsubscribe(symbol)

    del active_positions[symbol]
    return pnl


# =============================================================================
# ENTRY LOGIC
# =============================================================================

def scan_for_entry():
    """Scan watchlist for entry opportunities."""
    global watchlist, last_watchlist_refresh

    if not can_place_new_trade():
        return

    # Refresh watchlist periodically
    now = datetime.now()
    if (last_watchlist_refresh is None or
        (now - last_watchlist_refresh).total_seconds() > WATCHLIST_REFRESH_INTERVAL):

        print("\n" + "=" * 80)
        print(f"🔄 REFRESHING WATCHLIST - 15 minutes elapsed since last refresh")
        print("=" * 80)

        watchlist = get_dynamic_watchlist()
        last_watchlist_refresh = now
        print(f"✅ Watchlist refreshed: {len(watchlist)} symbols\n")

    print_scan_header()

    # Process in batches
    batch_num = 0
    for i in range(0, len(watchlist), BATCH_SIZE):
        batch = watchlist[i:i + BATCH_SIZE]
        batch_num += 1

        print(f"\n📦 Processing batch {batch_num}: {batch}")

        for symbol in batch:
            if symbol in active_positions:
                continue

            try:
                # Rate limit API calls
                data_limiter.wait_if_needed()

                # Fetch historical data
                fetch_time = datetime.now()
                print(f"[DATA_API] ✅ Fetching data for '{symbol}' at {fetch_time.strftime('%H:%M:%S')}")

                df = tsl.get_candles(symbol, tsl.NSE, "5m", 100)

                if df is None or df.empty:
                    print(f"[ERROR] Failed to fetch historical data for {symbol}. Returned None.")
                    print(f"⏭️  Skipping {symbol} - No chart data")
                    continue

                if len(df) < 50:
                    print(f"⏭️  Skipping {symbol} - Insufficient data ({len(df)} candles)")
                    continue

                complete_time = datetime.now()
                print(f"✅ Completed fetching for {symbol} at {complete_time.strftime('%H:%M:%S')}")

                # Calculate indicators
                df = calculate_vwap(df)
                df = calculate_atr(df)
                df = calculate_adx(df)
                df = calculate_fractal_chaos_bands(df)

                # Check CE entry conditions
                ce_signal = check_ce_entry_conditions(df, symbol, tsl)
                if ce_signal["valid"]:
                    execute_ce_trade(symbol, df, ce_signal)
                    if not can_place_new_trade():
                        return

                # Check PE entry conditions
                pe_signal = check_pe_entry_conditions(df, symbol, tsl)
                if pe_signal["valid"]:
                    execute_pe_trade(symbol, df, pe_signal)
                    if not can_place_new_trade():
                        return

            except Exception as e:
                print(f"[ERROR] Error scanning {symbol}: {e}")
                traceback.print_exc()
                continue

        # Small delay between batches
        time.sleep(1)


def execute_ce_trade(symbol: str, df: pd.DataFrame, signal: Dict) -> None:
    """Execute a CE (Call) trade."""
    try:
        print(f"\n🟢 CE Entry Signal for {symbol}")

        result = execute_ce_entry(
            tsl=tsl,
            symbol=symbol,
            df=df,
            signal=signal,
            risk_per_trade=RISK_PER_TRADE,
            atr_multiplier=ATR_MULTIPLIER,
            option_sl_pct=OPTION_SL_PERCENTAGE
        )

        if result["success"]:
            add_position(
                symbol=result["option_symbol"],
                order_id=result["order_id"],
                entry_price=result["entry_price"],
                quantity=result["quantity"],
                option_type="CE",
                stop_loss=result["stop_loss"],
                target=result["target"]
            )

            # Subscribe to WebSocket for live updates
            if ws_manager:
                ws_manager.subscribe(result["option_symbol"])

    except Exception as e:
        print(f"[ERROR] CE trade execution error: {e}")
        traceback.print_exc()


def execute_pe_trade(symbol: str, df: pd.DataFrame, signal: Dict) -> None:
    """Execute a PE (Put) trade."""
    try:
        print(f"\n🔴 PE Entry Signal for {symbol}")

        result = execute_pe_entry(
            tsl=tsl,
            symbol=symbol,
            df=df,
            signal=signal,
            risk_per_trade=RISK_PER_TRADE,
            atr_multiplier=ATR_MULTIPLIER,
            option_sl_pct=OPTION_SL_PERCENTAGE
        )

        if result["success"]:
            add_position(
                symbol=result["option_symbol"],
                order_id=result["order_id"],
                entry_price=result["entry_price"],
                quantity=result["quantity"],
                option_type="PE",
                stop_loss=result["stop_loss"],
                target=result["target"]
            )

            # Subscribe to WebSocket for live updates
            if ws_manager:
                ws_manager.subscribe(result["option_symbol"])

    except Exception as e:
        print(f"[ERROR] PE trade execution error: {e}")
        traceback.print_exc()


# =============================================================================
# EXIT LOGIC
# =============================================================================

def monitor_positions():
    """Monitor active positions for exit conditions."""
    if not active_positions:
        return

    for symbol, position in list(active_positions.items()):
        try:
            print_monitor_header(symbol)

            # Get current price
            current_price = ws_manager.get_ltp(symbol) if ws_manager else 0

            if current_price == 0:
                data_limiter.wait_if_needed()
                current_price = tsl.get_ltp(symbol, tsl.NFO)

            if current_price == 0:
                print(f"⚠️ Could not get price for {symbol}")
                continue

            entry_price = position["entry_price"]
            unrealized_pnl = (current_price - entry_price) * position["quantity"]
            pnl_pct = ((current_price / entry_price) - 1) * 100

            print(f"💵 Entry: ₹{entry_price:.2f} | Current: ₹{current_price:.2f}")
            print(f"📈 Unrealized P&L: ₹{unrealized_pnl:,.2f} ({pnl_pct:.1f}%)")
            print(f"🛑 SL: ₹{position['stop_loss']:.2f} | TSL: ₹{position['trailing_stop']:.2f}")

            # Update high/low tracking
            position["highest_price"] = max(position["highest_price"], current_price)
            position["lowest_price"] = min(position["lowest_price"], current_price)

            # Update current price in trade logger
            if trade_logger:
                trade_logger.update_current_price(symbol, current_price)

            # Check exit conditions
            exit_signal = check_exit_conditions(
                position=position,
                current_price=current_price,
                option_type=position["option_type"]
            )

            if exit_signal["should_exit"]:
                execute_position_exit(symbol, current_price, exit_signal["reason"])
            else:
                # Update trailing stop if applicable
                new_tsl = update_trailing_stop(
                    position=position,
                    current_price=current_price,
                    atr_multiplier=ATR_MULTIPLIER
                )
                if new_tsl > position["trailing_stop"]:
                    old_tsl = position["trailing_stop"]
                    position["trailing_stop"] = new_tsl
                    print(f"📈 TSL Updated: ₹{old_tsl:.2f} → ₹{new_tsl:.2f}")
                    if trade_logger:
                        trade_logger.log_tsl_update(symbol, old_tsl, new_tsl)

        except Exception as e:
            print(f"[ERROR] Error monitoring {symbol}: {e}")
            traceback.print_exc()


def execute_position_exit(symbol: str, exit_price: float, reason: str) -> None:
    """Execute position exit."""
    try:
        position = active_positions[symbol]

        result = execute_exit(
            tsl=tsl,
            symbol=symbol,
            quantity=position["quantity"],
            exit_price=exit_price
        )

        if result["success"]:
            remove_position(symbol, result["executed_price"], reason)
        else:
            print(f"[ERROR] Exit failed for {symbol}: {result['message']}")

    except Exception as e:
        print(f"[ERROR] Exit execution error: {e}")
        traceback.print_exc()


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def run_trading_loop():
    """Main trading loop."""
    print_bot_started_banner()

    last_scan_time = None
    last_monitor_time = None
    last_market_check = None

    while True:
        try:
            now = datetime.now()

            # Print market status periodically
            if last_market_check is None or (now - last_market_check).total_seconds() >= 300:
                print_market_status()
                last_market_check = now

            # Check market hours
            if not is_market_hours():
                if now.hour >= MARKET_CLOSE_HOUR:
                    print("\n" + "=" * 80)
                    print("🔔 MARKET CLOSED - Ending session")
                    print("=" * 80)
                    break
                else:
                    print(f"⏳ Waiting for market to open... ({now.strftime('%H:%M:%S')})")
                    time.sleep(60)
                    continue

            # Monitor existing positions
            if active_positions:
                if (last_monitor_time is None or
                    (now - last_monitor_time).total_seconds() >= MONITOR_INTERVAL):
                    monitor_positions()
                    last_monitor_time = now

            # Scan for new entries
            if can_place_new_trade():
                if (last_scan_time is None or
                    (now - last_scan_time).total_seconds() >= SCAN_INTERVAL):
                    scan_for_entry()
                    last_scan_time = now

            # Short sleep to prevent CPU overload
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n⚠️ Bot stopped by user (Ctrl+C)\n")
            break
        except Exception as e:
            print(f"[ERROR] Error in main loop: {e}")
            traceback.print_exc()
            time.sleep(5)

    # End of day summary
    print_session_summary()


def print_session_summary():
    """Print end of session summary."""
    print("\n" + "=" * 80)
    print("📊 SESSION SUMMARY")
    print("=" * 80)
    print(f"📋 Total Orders: {todays_orders}")
    pnl_emoji = "💰" if todays_pnl >= 0 else "📉"
    print(f"{pnl_emoji} Total P&L: ₹{todays_pnl:,.2f}")
    print(f"👁️  Open Positions: {len(active_positions)}")

    if active_positions:
        print("\n📈 Open Positions:")
        for symbol, pos in active_positions.items():
            print(f"   {symbol}: Entry=₹{pos['entry_price']:.2f}, TSL=₹{pos['trailing_stop']:.2f}")

    print("=" * 80)

    # Print trade logger summary with Excel file location
    if trade_logger:
        trade_logger.print_summary()
        trade_logger.close_excel()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    print_startup_banner()

    try:
        # Initialize
        if not initialize_bot():
            print("[ERROR] Bot initialization failed!")
            return

        # Run trading loop
        run_trading_loop()

    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
