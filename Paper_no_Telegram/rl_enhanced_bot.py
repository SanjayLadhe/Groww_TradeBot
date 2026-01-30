"""
RL-Enhanced Trading Bot

This module integrates the RL agent with your existing trading strategy.
The RL acts as a meta-layer that optimizes:
1. Entry decisions (take signal or skip)
2. Position sizing (lot multiplier)
3. Risk parameters (SL/Target adjustments)
4. Exit timing (hold vs exit)

Your original strategy (VWAP, ADX, ATR, Fractal) generates signals.
The RL agent decides whether and how to act on those signals.

Usage:
    python rl_enhanced_bot.py

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

# Entry/Exit Logic Modules (your existing strategy)
from ce_entry_logic import check_ce_entry_conditions, calculate_position_size, get_option_details
from pe_entry_logic import check_pe_entry_conditions
from exit_logic import check_exit_conditions, execute_exit, update_trailing_stop

# Technical Indicators (your existing indicators)
from VWAP import calculate_vwap
from ATRTrailingStop import calculate_atr, calculate_atr_trailing_stop
from adx_indicator import calculate_adx
from Fractal_Chaos_Bands import calculate_fractal_chaos_bands

# Utility Modules
from rate_limiter import RateLimiter, get_data_api_limiter, get_order_api_limiter
from websocket_manager import WebSocketManager
from SectorPerformanceAnalyzer import get_dynamic_watchlist

# Paper Trading
from paper_trading_config import PAPER_TRADING_ENABLED, PAPER_TRADING_BALANCE

# Trade Logger
from trade_logger import TradeLogger, init_trade_logger

# RL Components
from rl_config import (
    RL_ENABLED,
    USE_PRETRAINED_MODEL,
    PRETRAINED_MODEL_PATH,
    HARD_LIMITS,
    MIN_CONFIDENCE_TO_ACT
)
from rl_agent import RLDecisionMaker
from rl_reward import RewardCalculator, calculate_trade_reward
from rl_state_builder import StateBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Credentials
API_KEY = "YOUR_GROWW_API_KEY"
API_SECRET = "YOUR_GROWW_API_SECRET"

# Trading Parameters (base values - RL will adjust these)
OPENING_BALANCE = 1005000
BASE_CAPITAL = 1000000
MARKET_MONEY_RISK_PCT = 0.01
BASE_CAPITAL_RISK_PCT = 0.005

MAX_ORDERS_TODAY = 2
MAX_SIMULTANEOUS_POSITIONS = 2

RISK_PER_TRADE = None  # Calculated dynamically

# Base risk parameters (RL will adjust these)
BASE_ATR_MULTIPLIER = 3
BASE_RISK_REWARD_RATIO = 3
BASE_OPTION_SL_PERCENTAGE = 0.15

# Timing
MONITOR_INTERVAL = 5
SCAN_INTERVAL = 30
WATCHLIST_REFRESH_INTERVAL = 900
BATCH_SIZE = 5

# Market Hours
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
NO_NEW_TRADE_HOUR = 14
NO_NEW_TRADE_MINUTE = 45

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

tsl = None
active_positions: Dict[str, Dict] = {}
todays_orders = 0
todays_pnl = 0.0
todays_wins = 0
todays_losses = 0
watchlist: List[str] = []
last_watchlist_refresh = None
ws_manager = None
data_limiter = None
order_limiter = None
trade_logger: TradeLogger = None

# RL Components
rl_decision_maker: RLDecisionMaker = None
reward_calculator: RewardCalculator = None

# =============================================================================
# BANNER AND DISPLAY
# =============================================================================

def print_startup_banner():
    """Print startup banner with RL mode info."""
    print("\n" + "=" * 100)
    print("🤖 RL-ENHANCED GROWW TRADING BOT")
    print("=" * 100)
    
    if RL_ENABLED:
        print("🧠 REINFORCEMENT LEARNING: ENABLED")
        print("   - RL optimizes entry decisions, position sizing, and risk parameters")
        print("   - Your base strategy (VWAP, ADX, ATR, Fractal) generates signals")
        print("   - RL decides whether and how to act on those signals")
        if USE_PRETRAINED_MODEL:
            print(f"   - Using pre-trained model: {PRETRAINED_MODEL_PATH}")
        else:
            print("   - Training from scratch (online learning)")
    else:
        print("🧠 REINFORCEMENT LEARNING: DISABLED")
        print("   - Running original strategy without RL optimization")
    
    print("-" * 100)
    
    if PAPER_TRADING_ENABLED:
        print("🎮 PAPER TRADING MODE")
        print(f"💰 Starting Balance: ₹{PAPER_TRADING_BALANCE:,.2f}")
    else:
        print("🚀 LIVE TRADING MODE")
        print(f"💰 Opening Balance: ₹{OPENING_BALANCE:,.2f}")
    
    print("=" * 100 + "\n")


def print_rl_decision(symbol: str, signal_type: str, rl_action: Dict):
    """Print RL decision details."""
    if not RL_ENABLED:
        return
    
    take = "✅ TAKE" if rl_action["take_signal"] else "❌ SKIP"
    conf = rl_action["confidence"] * 100
    
    print(f"\n{'=' * 70}")
    print(f"🧠 RL DECISION for {symbol} ({signal_type})")
    print(f"{'=' * 70}")
    print(f"   Decision:      {take}")
    print(f"   Confidence:    {conf:.1f}%")
    print(f"   Size Mult:     {rl_action['position_size_multiplier']:.2f}x")
    print(f"   SL Adjust:     {rl_action['sl_adjustment']:.2f}x")
    print(f"   Target Adjust: {rl_action['target_adjustment']:.2f}x")
    print(f"   TSL Factor:    {rl_action['trailing_stop_factor']:.2f}x")
    print(f"{'=' * 70}")


def print_bot_started_banner():
    """Print bot started banner."""
    print("\n" + "=" * 100)
    print("🚀 RL-ENHANCED BOT STARTED")
    print("=" * 100)
    print(f"📊 Strategy: Your indicators + RL optimization")
    print(f"⏰ Scan Interval: {SCAN_INTERVAL}s | Monitor: {MONITOR_INTERVAL}s")
    print(f"💰 Risk per Trade: ₹{RISK_PER_TRADE:,.2f}")
    print(f"📋 Watchlist: {len(watchlist)} symbols")
    print(f"🧠 RL Confidence Threshold: {MIN_CONFIDENCE_TO_ACT * 100:.0f}%")
    print("=" * 100 + "\n")


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_bot():
    """Initialize the RL-enhanced trading bot."""
    global tsl, ws_manager, data_limiter, order_limiter, watchlist
    global RISK_PER_TRADE, trade_logger
    global rl_decision_maker, reward_calculator
    
    print("\n🤖 Groww RL-Enhanced Bot Version 1.0")
    print("-----Initializing Components-----")
    
    # Initialize trade logger
    trade_logger = init_trade_logger(log_dir="logs", excel_file="RL_Trade_Data.xlsx")
    
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

    # Connect paper trading simulator to WebSocket manager for price synchronization
    if PAPER_TRADING_ENABLED and hasattr(tsl, 'set_ws_manager'):
        tsl.set_ws_manager(ws_manager)

    # Initialize RL components
    if RL_ENABLED:
        print("\n🧠 Initializing RL Components...")
        rl_decision_maker = RLDecisionMaker(
            load_pretrained=USE_PRETRAINED_MODEL,
            model_path=PRETRAINED_MODEL_PATH if USE_PRETRAINED_MODEL else None
        )
        reward_calculator = RewardCalculator()
        print("✅ RL Decision Maker initialized")
    
    # Get initial watchlist
    print("\nFetching watchlist from best performing sectors...")
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


def get_session_stats() -> Dict:
    """Get current session statistics for RL state."""
    win_rate = (todays_wins / (todays_wins + todays_losses) * 100) if (todays_wins + todays_losses) > 0 else 0
    return {
        "daily_pnl": todays_pnl,
        "win_rate": win_rate,
        "trades_today": todays_orders,
        "max_trades": MAX_ORDERS_TODAY
    }


# =============================================================================
# RL-ENHANCED POSITION MANAGEMENT
# =============================================================================

def add_position(
    symbol: str,
    order_id: str,
    entry_price: float,
    quantity: int,
    option_type: str,
    stop_loss: float,
    target: float,
    rl_action: Dict = None
) -> None:
    """Add a new position with RL parameters stored."""
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
        "lowest_price": entry_price,
        "current_price": entry_price,
        # RL-specific tracking
        "rl_action": rl_action,
        "rl_confidence": rl_action.get("confidence", 1.0) if rl_action else 1.0,
        "original_sl": stop_loss,
        "original_target": target
    }
    
    todays_orders += 1
    
    # Log to trade logger
    if trade_logger:
        trade_logger.log_order(
            order_id=order_id, symbol=symbol, transaction_type="BUY",
            quantity=quantity, order_type="MARKET", price=entry_price,
            status="COMPLETE", option_type=option_type,
            rl_confidence=rl_action.get("confidence", 0) if rl_action else 0
        )
        trade_logger.log_position_open(
            symbol=symbol, entry_price=entry_price, quantity=quantity,
            stop_loss=stop_loss, target=target, option_type=option_type
        )


def remove_position(symbol: str, exit_price: float, reason: str) -> float:
    """Remove position and provide reward to RL agent."""
    global todays_pnl, todays_wins, todays_losses
    
    if symbol not in active_positions:
        return 0.0
    
    position = active_positions[symbol]
    entry_price = position["entry_price"]
    quantity = position["quantity"]
    entry_time = position["entry_time"]
    
    pnl = (exit_price - entry_price) * quantity
    pnl_pct = ((exit_price / entry_price) - 1) * 100
    todays_pnl += pnl
    
    if pnl >= 0:
        todays_wins += 1
    else:
        todays_losses += 1
    
    # Calculate time in trade
    time_in_trade = (datetime.now() - entry_time).total_seconds() / 60
    
    # Calculate RL reward
    if RL_ENABLED and rl_decision_maker:
        trade_result = {
            "action_taken": True,
            "position_closed": True,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "stop_loss": position["stop_loss"],
            "target": position["target"],
            "exit_reason": reason,
            "time_in_trade": time_in_trade
        }
        
        reward = calculate_trade_reward(trade_result)
        rl_decision_maker.learn_from_trade(reward, done=False)
        
        print(f"\n🧠 RL Reward: {reward:.4f} (P&L: ₹{pnl:,.2f}, {pnl_pct:.1f}%)")
    
    # Log to trade logger
    if trade_logger:
        trade_logger.log_position_close(symbol, exit_price, reason)
    
    # Unsubscribe from WebSocket
    if ws_manager:
        ws_manager.unsubscribe(symbol)
    
    del active_positions[symbol]
    return pnl


# =============================================================================
# RL-ENHANCED ENTRY LOGIC
# =============================================================================

def scan_for_entry():
    """Scan watchlist with RL-enhanced decision making."""
    global watchlist, last_watchlist_refresh
    
    if not can_place_new_trade():
        return
    
    # Refresh watchlist periodically
    now = datetime.now()
    if (last_watchlist_refresh is None or
        (now - last_watchlist_refresh).total_seconds() > WATCHLIST_REFRESH_INTERVAL):
        watchlist = get_dynamic_watchlist()
        last_watchlist_refresh = now
        print(f"\n🔄 Watchlist refreshed: {len(watchlist)} symbols")
    
    print(f"\n{'=' * 80}")
    print(f"🔍 SCANNING FOR ENTRY - {now.strftime('%H:%M:%S')}")
    print(f"{'=' * 80}")
    
    session_stats = get_session_stats()
    
    for i in range(0, len(watchlist), BATCH_SIZE):
        batch = watchlist[i:i + BATCH_SIZE]
        
        for symbol in batch:
            if symbol in active_positions:
                continue
            
            try:
                data_limiter.wait_if_needed()
                df = tsl.get_candles(symbol, tsl.NSE, "5m", 100)
                
                if df is None or df.empty or len(df) < 50:
                    continue
                
                # Calculate indicators (your existing logic)
                df = calculate_vwap(df)
                df = calculate_atr(df)
                df = calculate_adx(df)
                df = calculate_fractal_chaos_bands(df)
                
                # Check CE conditions (your existing logic)
                ce_signal = check_ce_entry_conditions(df, symbol, tsl)
                if ce_signal["valid"]:
                    execute_rl_enhanced_trade(symbol, df, ce_signal, "CE", session_stats)
                    if not can_place_new_trade():
                        return
                
                # Check PE conditions (your existing logic)
                pe_signal = check_pe_entry_conditions(df, symbol, tsl)
                if pe_signal["valid"]:
                    execute_rl_enhanced_trade(symbol, df, pe_signal, "PE", session_stats)
                    if not can_place_new_trade():
                        return
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        time.sleep(1)


def execute_rl_enhanced_trade(
    symbol: str,
    df: pd.DataFrame,
    signal: Dict,
    option_type: str,
    session_stats: Dict
) -> None:
    """Execute trade with RL-enhanced parameters."""
    try:
        print(f"\n{'🟢' if option_type == 'CE' else '🔴'} {option_type} Signal detected for {symbol}")
        
        # Get RL decision
        if RL_ENABLED and rl_decision_maker:
            rl_action = rl_decision_maker.should_take_signal(
                df, option_type, None, session_stats
            )
            print_rl_decision(symbol, option_type, rl_action)
            
            # Check if RL says to skip
            if not rl_action["take_signal"]:
                print(f"🧠 RL Decision: SKIP signal (confidence: {rl_action['confidence']*100:.1f}%)")
                
                # Learn from skipped signal (small negative reward if signal would have been good)
                rl_decision_maker.learn_from_trade(reward=-0.01, done=False)
                return
        else:
            rl_action = {
                "take_signal": True,
                "confidence": 1.0,
                "position_size_multiplier": 1.0,
                "sl_adjustment": 1.0,
                "target_adjustment": 1.0,
                "trailing_stop_factor": 1.0
            }
        
        # Get base option details
        spot_price = signal['close']
        option_info = get_option_details(tsl, symbol, spot_price, option_type)
        
        if not option_info["valid"]:
            return
        
        option_symbol = option_info["option_symbol"]
        entry_price = option_info["ltp"]
        lot_size = option_info["lot_size"]
        
        if entry_price <= 0:
            return
        
        # Apply RL adjustments to parameters
        adjusted_sl_pct = BASE_OPTION_SL_PERCENTAGE * rl_action["sl_adjustment"]
        adjusted_sl_pct = np.clip(adjusted_sl_pct, 
                                   HARD_LIMITS["min_stop_loss_pct"],
                                   HARD_LIMITS["max_stop_loss_pct"])
        
        # Calculate stop loss and target
        stop_loss = entry_price * (1 - adjusted_sl_pct)
        risk = entry_price - stop_loss
        
        adjusted_rr = BASE_RISK_REWARD_RATIO * rl_action["target_adjustment"]
        target = entry_price + (risk * adjusted_rr)
        
        # Calculate position size with RL adjustment
        base_lots = calculate_position_size(entry_price, stop_loss, RISK_PER_TRADE, lot_size)
        adjusted_lots = int(base_lots * rl_action["position_size_multiplier"])
        adjusted_lots = max(1, min(adjusted_lots, HARD_LIMITS["max_position_size"]))
        quantity = adjusted_lots * lot_size
        
        print(f"\n📊 RL-Adjusted Trade Parameters:")
        print(f"   Entry: ₹{entry_price:.2f}")
        print(f"   SL: ₹{stop_loss:.2f} ({adjusted_sl_pct*100:.1f}%)")
        print(f"   Target: ₹{target:.2f} (RR: 1:{adjusted_rr:.1f})")
        print(f"   Lots: {adjusted_lots} (base: {base_lots})")
        
        # Place order (pass entry_price so simulator uses consistent price)
        order_result = tsl.place_order(
            symbol=option_symbol,
            exchange=tsl.NFO,
            transaction_type=tsl.BUY,
            quantity=quantity,
            order_type=tsl.ORDER_TYPE_MARKET,
            price=entry_price,
            product_type=tsl.PRODUCT_INTRADAY
        )
        
        if order_result["status"] == "success":
            order_id = order_result["order_id"]
            order_status = tsl.get_order_status(order_id)
            executed_price = order_status.get("average_price", entry_price)
            
            # Store RL parameters with position
            add_position(
                symbol=option_symbol,
                order_id=order_id,
                entry_price=executed_price,
                quantity=quantity,
                option_type=option_type,
                stop_loss=stop_loss,
                target=target,
                rl_action=rl_action
            )
            
            # Subscribe to WebSocket
            if ws_manager:
                ws_manager.subscribe(option_symbol)
            
            print(f"\n✅ {option_type} Order Executed: {order_id}")
            
    except Exception as e:
        logger.error(f"RL-enhanced trade error: {e}")
        traceback.print_exc()


# =============================================================================
# RL-ENHANCED EXIT LOGIC
# =============================================================================

def monitor_positions():
    """Monitor positions with RL-enhanced exit decisions."""
    if not active_positions:
        return
    
    for symbol, position in list(active_positions.items()):
        try:
            now = datetime.now()
            print(f"\n{'=' * 60}")
            print(f"👁️ MONITORING: {symbol} - {now.strftime('%H:%M:%S')}")
            
            # Get current price
            current_price = ws_manager.get_ltp(symbol) if ws_manager else 0
            if current_price == 0:
                data_limiter.wait_if_needed()
                current_price = tsl.get_ltp(symbol, tsl.NFO)
            
            if current_price == 0:
                continue
            
            position["current_price"] = current_price
            entry_price = position["entry_price"]
            unrealized_pnl = (current_price - entry_price) * position["quantity"]
            pnl_pct = ((current_price / entry_price) - 1) * 100
            
            print(f"💵 Entry: ₹{entry_price:.2f} | Current: ₹{current_price:.2f}")
            print(f"📈 P&L: ₹{unrealized_pnl:,.2f} ({pnl_pct:.1f}%)")
            
            # Update high/low tracking
            position["highest_price"] = max(position["highest_price"], current_price)
            position["lowest_price"] = min(position["lowest_price"], current_price)
            
            # Update trade logger
            if trade_logger:
                trade_logger.update_current_price(symbol, current_price)
            
            # Check standard exit conditions (your existing logic)
            exit_signal = check_exit_conditions(
                position=position,
                current_price=current_price,
                option_type=position["option_type"]
            )
            
            # RL can also suggest early exit
            if RL_ENABLED and rl_decision_maker and not exit_signal["should_exit"]:
                session_stats = get_session_stats()
                
                # Fetch fresh data for RL decision
                try:
                    df = tsl.get_candles(symbol.split()[0], tsl.NSE, "5m", 50)
                    if df is not None and not df.empty:
                        df = calculate_vwap(df)
                        df = calculate_atr(df)
                        df = calculate_adx(df)
                        
                        rl_exit = rl_decision_maker.should_exit_position(
                            df, position, current_price, session_stats
                        )
                        
                        if rl_exit["should_exit"]:
                            print(f"🧠 RL suggests exit (confidence: {rl_exit['confidence']*100:.1f}%)")
                            exit_signal = {
                                "should_exit": True,
                                "reason": "RL_EXIT",
                                "exit_type": "RL"
                            }
                except:
                    pass
            
            if exit_signal["should_exit"]:
                execute_position_exit(symbol, current_price, exit_signal["reason"])
            else:
                # Update trailing stop with RL adjustment
                rl_tsl_factor = position.get("rl_action", {}).get("trailing_stop_factor", 1.0)
                new_tsl = update_trailing_stop(
                    position=position,
                    current_price=current_price,
                    atr_multiplier=BASE_ATR_MULTIPLIER * rl_tsl_factor
                )
                
                if new_tsl > position["trailing_stop"]:
                    old_tsl = position["trailing_stop"]
                    position["trailing_stop"] = new_tsl
                    print(f"📈 TSL Updated: ₹{old_tsl:.2f} → ₹{new_tsl:.2f}")
                    
                    if trade_logger:
                        trade_logger.log_tsl_update(symbol, old_tsl, new_tsl)
                
                # Provide intermediate reward to RL
                if RL_ENABLED and rl_decision_maker:
                    time_in_trade = (now - position["entry_time"]).total_seconds() / 60
                    intermediate_reward = reward_calculator.calculate_reward(
                        action_taken=True,
                        position_opened=False,
                        position_closed=False,
                        entry_price=entry_price,
                        current_price=current_price,
                        stop_loss=position["stop_loss"],
                        target=position["target"],
                        quantity=position["quantity"],
                        time_in_trade=time_in_trade
                    )
                    # Small learning step for holding
                    if abs(intermediate_reward) > 0.01:
                        rl_decision_maker.learn_from_trade(intermediate_reward * 0.1, done=False)
        
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")
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
            pnl = remove_position(symbol, result["executed_price"], reason)
            
            emoji = "💰" if pnl >= 0 else "📉"
            print(f"\n{emoji} Position closed: {reason}")
            print(f"   P&L: ₹{pnl:,.2f}")
        
    except Exception as e:
        logger.error(f"Exit execution error: {e}")


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def run_trading_loop():
    """Main trading loop with RL integration."""
    print_bot_started_banner()
    
    last_scan_time = None
    last_monitor_time = None
    last_save_time = datetime.now()
    
    while True:
        try:
            now = datetime.now()
            
            # Check market hours
            if not is_market_hours():
                if now.hour >= MARKET_CLOSE_HOUR:
                    print("\n🔔 MARKET CLOSED - Ending session")
                    break
                else:
                    print(f"⏳ Waiting for market... ({now.strftime('%H:%M:%S')})")
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
            
            # Periodically save RL model
            if RL_ENABLED and rl_decision_maker:
                if (now - last_save_time).total_seconds() >= 1800:  # Every 30 min
                    rl_decision_maker.save_model()
                    last_save_time = now
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            traceback.print_exc()
            time.sleep(5)
    
    # End of day
    print_session_summary()
    
    # Final model save
    if RL_ENABLED and rl_decision_maker:
        rl_decision_maker.save_model()
        print("🧠 RL model saved")


def print_session_summary():
    """Print session summary with RL stats."""
    print("\n" + "=" * 80)
    print("📊 SESSION SUMMARY")
    print("=" * 80)
    
    print(f"📋 Total Orders: {todays_orders}")
    print(f"✅ Wins: {todays_wins} | ❌ Losses: {todays_losses}")
    
    win_rate = (todays_wins / (todays_wins + todays_losses) * 100) if (todays_wins + todays_losses) > 0 else 0
    print(f"📈 Win Rate: {win_rate:.1f}%")
    
    pnl_emoji = "💰" if todays_pnl >= 0 else "📉"
    print(f"{pnl_emoji} Total P&L: ₹{todays_pnl:,.2f}")
    
    if active_positions:
        print(f"\n👁️ Open Positions: {len(active_positions)}")
    
    if RL_ENABLED:
        print("\n🧠 RL Training Stats:")
        print(f"   Training Steps: {rl_decision_maker.agent.training_step}")
    
    print("=" * 80)
    
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
        if not initialize_bot():
            print("[ERROR] Bot initialization failed!")
            return
        
        run_trading_loop()
        
    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
